# Notatki: matmul-bench

Pojęcia, wzory i obserwacje z pomiarów, etap po etapie.

---

## Etap 0: sprzęt, teoretyczny szczyt, metodologia

### Sprzęt

| | |
|---|---|
| CPU | Intel Core i7-13620H (Raptor Lake, architektura hybrydowa) |
| Rdzenie | 6 × P-core (Raptor Cove, z HT, czyli 12 wątków) + 4 × E-core (Gracemont, bez HT) = 10 rdzeni / 16 wątków |
| Zegary (specyfikacja Intela) | P: 2,4 GHz bazowo / 4,9 GHz turbo; E: 1,8 / 3,6 GHz |
| L1d | P: 48 KiB na rdzeń, E: 32 KiB na rdzeń |
| L2 | P: 1,25 MiB na rdzeń, E: 2 MiB na klaster 4 rdzeni (razem 9,5 MiB) |
| L3 | 24 MiB, wspólne dla wszystkich rdzeni |
| SIMD | AVX2 + FMA (rejestr 256 bit = 8 floatów), **bez AVX-512** |
| RAM | 2 × 32 GB DDR4-3200, dwa kanały: teoretycznie 2 × 8 B × 3200 MT/s = **51,2 GB/s** |
| Środowisko | Windows 11 + WSL2 (Ubuntu 24.04), g++ 13.3, CMake 3.28, Ninja 1.11 |
| Flagi | `-O3 -DNDEBUG -march=native`, bez `-ffast-math` |

Linia cache ma 64 B, czyli 16 floatów. To najmniejsza porcja, jaką pamięć przesyła do cache.

### WSL2 pokazuje fałszywą topologię

`lscpu` w WSL2 pokazuje „8 rdzeni × 2 wątki” i 8 jednakowych L2 po 1,25 MiB. To topologia maszyny
wirtualnej (Hyper-V), a nie prawdziwego procesora. Prawdziwą widać po stronie Windows
(`Win32_Processor`: 10 rdzeni, 16 procesorów logicznych, L2 = 9728 KB). Google Benchmark też czyta
tę wirtualną topologię. Pokazywane przez niego „2918 MHz” to częstotliwość licznika TSC, a nie
faktyczny zegar rdzenia.

Konsekwencje:
- Linux w WSL nie wie, który vCPU trafia na rdzeń P, a który na E. Decyduje planista Windows,
  więc `taskset` przypina wątek do vCPU, ale nie do fizycznego rdzenia.
- Pomiar jednowątkowy może trafić raz na rdzeń P, raz na E. To daje dwie grupy wyników.
- Przy skalowaniu wątków (etap 4) ważne punkty to 6 (wszystkie P), 10 (wszystkie rdzenie
  fizyczne) i 16 (z HT).

### Teoretyczny szczyt FP32

```
szczyt = rdzenie × zegar × (floatów w rejestrze SIMD) × (instrukcji FMA na cykl) × 2
```

- **× 2**, bo FMA (`a*b + c`) to dwie operacje zmiennoprzecinkowe w jednej instrukcji.
- **Rdzeń P**: dwie jednostki FMA 256-bit, więc 2 × 8 × 2 = **32 FLOP/cykl**.
- **Rdzeń E**: dwa potoki 128-bit. Instrukcja 256-bit jest dzielona na dwie połówki, co daje
  równowartość jednej FMA 256-bit na cykl: **16 FLOP/cykl**.
- **Hyperthreading nie podnosi szczytu.** Dwa wątki sprzętowe jednego rdzenia dzielą te same
  jednostki FMA. HT pomaga tylko wtedy, gdy jeden wątek zostawia jednostki bezczynne, np. czekając
  na pamięć.

| Scenariusz | Obliczenie | GFLOP/s |
|---|---|---|
| 1 rdzeń P, turbo 4,9 GHz | 32 × 4,9 | **156,8** |
| 1 rdzeń E, turbo 3,6 GHz | 16 × 3,6 | 57,6 |
| wszystkie rdzenie, zegary bazowe | 6·32·2,4 + 4·16·1,8 | 576 |
| wszystkie rdzenie, max turbo (w praktyce nieosiągalne) | 6·32·4,9 + 4·16·3,6 | 1171 |

Przy obciążeniu wszystkich rdzeni laptop ogranicza limit mocy (45 W bazowo, chwilowo do 115 W)
i chłodzenie. Realny szczyt wielowątkowy leży więc gdzieś w przedziale 576–1171 i trzeba go
zmierzyć, a nie założyć. **Punkt odniesienia dla wersji jednowątkowych (etapy 1–3): ~157 GFLOP/s.**

> **Aktualizacja z etapu 1:** zmierzony zegar rdzenia P przy obecnych ustawieniach zasilania to
> ok. 2,84 GHz, a nie 4,9 GHz. Realny szczyt jednego rdzenia wynosi więc 32 × 2,84 ≈ **91 GFLOP/s**.
> Szczegóły w sekcji etapu 1.

### GFLOP/s dla mnożenia macierzy

Dla C = A·B (N×N) każdy z N² elementów C jest iloczynem skalarnym długości N, czyli N mnożeń
i N dodawań. Razem:

```
FLOP = 2·N³            GFLOP/s = 2·N³ / (czas_w_sekundach · 10⁹)
```

| N | FLOP | czas przy 157 GFLOP/s |
|---|---|---|
| 256 | 33,6 M | 0,21 ms |
| 512 | 268 M | 1,7 ms |
| 1024 | 2,15 G | 13,7 ms |

### Arithmetic intensity: pierwsze podejście

**Arithmetic intensity (AI)** = liczba FLOP / liczba bajtów przesłanych z pamięci i do pamięci.

saxpy (`y = a·x + y`) na element: 2 FLOP, a ruch to odczyt x (4 B), odczyt y (4 B) i zapis y (4 B),
razem 12 B. AI = 2/12 ≈ **0,17 FLOP/B**. Żeby rdzeń P liczył z pełną szybkością przy takim AI,
potrzebowałby 32 / 0,17 ≈ 192 B danych na cykl. Tego nie daje żaden poziom pamięci, więc saxpy
jest ograniczony pamięcią (*memory-bound*). Jego GFLOP/s to w praktyce przepustowość pamięci
pomnożona przez 0,17.

### Pomiar: saxpy na kolejnych poziomach hierarchii (pomiar roboczy)

Warunki: plan zasilania „Zrównoważony”, zasilacz podłączony, w tle VS Code, 1 wątek,
5 powtórzeń, mediana. Hipoteza: kolejne poziomy hierarchii dają coraz mniej; L1 kilkadziesiąt
GFLOP/s, DRAM pojedyncze GFLOP/s.

| Zbiór roboczy | Mieści się w | Mediana GFLOP/s | Przepustowość | CV |
|---|---|---|---|---|
| 16 KiB | L1 | 24,4 (dwie grupy: ~24,5 i ~38) | 136 GiB/s | 22% |
| 512 KiB | L2 | 16,0 | 89 GiB/s | 7% |
| 8 MiB | L3 | 7,7 | 43 GiB/s | 3% |
| 128 MiB | DRAM | 2,1 | 11,7 GiB/s (12,6 GB/s) | 15% |

Drugi przebieg, z terminala przy mniejszym obciążeniu systemu (load average 0,29 zamiast 1,56):
L1 25,4 · L2 17,7 · L3 8,1 · DRAM 2,7 GFLOP/s (15,0 GiB/s). Kolejność poziomów jest ta sama.
Najbardziej na obciążenie w tle reaguje DRAM, bo pamięć RAM dzielą wszystkie programy
(także system Windows pod WSL). Oba przebiegi mają tam CV powyżej 10%.

Obserwacje:
- **Schodki są zgodne z hipotezą.** Między L1 a DRAM jest różnica ok. 12×, a licząc szybszą
  grupę w L1 ok. 18×. Te same obliczenia na tych samych danych, a różnica wynika wyłącznie z tego,
  gdzie leżą dane.
- **Nawet w L1 jesteśmy daleko od szczytu.** Najlepszy wynik 38 GFLOP/s to ok. 24% z 157.
  Ogranicza nas przepustowość instrukcji load/store i narzut krótkiej pętli, a nie jednostki FMA.
- **Jeden rdzeń nie nasyci pamięci RAM**: wyciąga ok. 12,6 GB/s, czyli mniej więcej 25%
  teoretycznych 51,2 GB/s. Wyjaśnia to **prawo Little'a**:
  przepustowość = (liczba żądań w locie) / opóźnienie. Rdzeń może czekać jednocześnie tylko na
  ograniczoną liczbę linii cache, a każda wędruje z RAM ok. 100 ns. Przykładowo
  16 linii × 64 B / 80 ns ≈ 12,8 GB/s, więc rząd wielkości się zgadza. Żeby nasycić pamięć,
  potrzeba kilku rdzeni naraz (wrócimy do tego w etapie 4).
- **L1 ma dwie grupy wyników** (~24,5 i ~38 GFLOP/s). Sekunda rozgrzewki niczego nie zmienia,
  więc to nie jest rozkręcanie się zegara. Śledztwo w trzech krokach:
  1. *Hipoteza: planista Windows przenosi wątek między rdzeniami P i E.* **Odrzucona.**
     Trzy niezależne przebiegi, w tym jeden z rozgrzewką, dały dokładnie ten sam wzór:
     wolne powtórzenia 1–3 i 6–9, szybkie 4–5 i 10. Losowe przerzucanie wątku nie powtarza się
     co do numeru powtórzenia, więc przyczyna jest deterministyczna.
  2. *Hipoteza: adresy buforów.* Każde powtórzenie alokuje `x` i `y` od nowa, a Google Benchmark
     między powtórzeniami też alokuje, więc sterta przesuwa się o wielokrotność 16 B.
     Benchmark diagnostyczny wypisywał adresy przy każdym powtórzeniu:

     | x mod 64 | y mod 64 | GFLOP/s |
     |---|---|---|
     | 48 | 0 | 32–36 |
     | 16 | 32 | 31–34 |
     | 0 | 16 | 22–23 |
     | 32 | 48 | 22 |

     Wolne są dokładnie te powtórzenia, w których `y` nie jest wyrównane do 32 B. Wtedy co drugi
     32-bajtowy zapis wektorowy przecina granicę linii cache (*split store*), więc procesor musi
     zapisać dwie linie zamiast jednej. Zapis przecinający linię kosztuje więcej niż taki sam
     odczyt: w najszybszym wierszu `x mod 64 = 48`, czyli co drugi odczyt `x` przecina linię,
     a wynik i tak jest dobry. Odpada też 4K aliasing jako wyjaśnienie podziału na grupy:
     `(y − x) mod 4096` wynosiło 16 we wszystkich powtórzeniach, także w szybkich.
  3. *Eksperyment kontrolowany:* `x` wyrównane do 4096 B, `y` przesunięte o zadaną liczbę
     bajtów. Najwolniejsze (25,8 GFLOP/s) jest przesunięcie 16 B, jedyne, przy którym `y` nie
     jest wyrównane do 32 B, co potwierdza krok 2. Przy wyrównanym `y` wyniki i tak wahają się od
     31 do 51 GFLOP/s w zależności od odległości `y − x`. To drugi efekt, prawdopodobnie fałszywe
     zależności zapis→odczyt przy zgodnych 12 najmłodszych bitach adresu (*4K aliasing*) albo
     konflikty w L1. Bez liczników sprzętowych, niedostępnych w WSL2, nie da się tego
     rozstrzygnąć, więc zostawiamy to jako pytanie otwarte.

  Wniosek: w mikrobenchmarku na danych z L1 sam adres bufora zmienia wynik nawet 2×
  (25,8 wobec 51,5 GFLOP/s). Dlatego w etapie 1 wszystkie bufory będą wyrównane do 64 B.
  Najlepszy zmierzony saxpy w L1 (51,5 GFLOP/s) to ok. 1/3 szczytu rdzenia.
- **Lekcje metodologiczne:**
  - Mediana jest odporna na pojedyncze odstające wyniki, ale potrafi ukryć dwie grupy.
    Zawsze patrzymy też na CV, a przy dużym CV na pojedyncze powtórzenia.
  - Wzór, który powtarza się co do numeru powtórzenia, oznacza przyczynę deterministyczną,
    a nie szum. Hipoteza obalona pomiarem to też wynik.

### Metodologia i pułapki pomiarowe

- Build Release (`-O3 -march=native`), **bez `-ffast-math`**, żeby kompilator nie zmieniał
  kolejności sumowania floatów.
- Stałe ziarno danych, co najmniej 5 powtórzeń, raportujemy medianę. **CV < 5%** to próg
  zaufania do wyniku.
- Przed pomiarami do README: zasilacz, tryb zasilania „Najlepsza wydajność”, zamknięte zbędne
  programy (przeglądarka, komunikatory), laptop na twardej powierzchni (chłodzenie). Puszczamy
  dwa przebiegi i porównujemy je ze sobą, bo przegrzanie obniża zegar.
- Turbo Boost: z WSL go nie wyłączymy, więc raportujemy uczciwie, że był włączony.
- Projekt leży na dysku Windows (`/mnt/c`), co spowalnia build (pierwsza konfiguracja ok. 2 min).
  Na wynik pomiaru nie ma to wpływu, bo dane programu są w RAM.

---

## Etap 1: wersja naiwna (i-j-k)

### Kod

```cpp
for i: for j: { sum = 0; for k: sum += A[i*n + k] * B[k*n + j]; C[i*n + j] = sum; }
```

`A[i][k]` idzie po wierszu, kolejne adresy co 4 B. `B[k][j]` idzie w dół kolumny, co `n·4` B.

### Dlaczego dostęp do B po kolumnach kosztuje

1. **Każdy element B to osobna linia cache.** Z 64 B linii używamy 4 B (6%). Pozostałe 15
   floatów z tej linii przydadzą się dopiero przy kolejnych `j`, o ile linia do tego czasu
   przetrwa w cache.
2. **Brak prawdziwej wektoryzacji.** Rejestr AVX ładuje 8 sąsiednich floatów, a elementy
   kolumny B leżą co `n` floatów. GCC zrobił półśrodek. Wczytuje 4 elementy kolumny pojedynczo
   i składa je w rejestr (ręczny *gather*), mnoży 4 naraz (`vmulps`), a potem dodaje 4 iloczyny
   do `sum` po kolei, czterema skalarnymi `vaddss`. Kolejność dodawania jest zachowana, więc
   wynik jest taki jak w kodzie źródłowym i nie trzeba `-ffast-math`. Ciekawostka: główna pętla
   używa osobnego mnożenia i dodawania (dwa zaokrąglenia), a pętla dla reszty `n mod 4` używa
   FMA (jedno zaokrąglenie).
3. **Konflikty w zbiorach cache przy potęgach dwójki.** Linia trafia do zbioru wyznaczonego
   przez bity 6–11 adresu w L1 (64 zbiory, 12 dróg) i bity 6–16 w L2 (2048 zbiorów, 10 dróg).
   Skok o 4 KiB nie zmienia bitów 6–11, więc cała kolumna B wpada do jednego zbioru L1.

   | N | skok | miejsca w L1 | miejsca w L2 | linii w kolumnie |
   |---|---|---|---|---|
   | 256 | 1 KiB | 4 zbiory × 12 = 48 | 128 × 10 = 1280 | 256 |
   | 512 | 2 KiB | 2 × 12 = 24 | 64 × 10 = 640 | 512 |
   | 1024 | 4 KiB | 1 × 12 = 12 | 32 × 10 = **320** | **1024** |
   | 1000 | ~3,9 KiB | rozłożone | rozłożone | 1000 |

4. **Prefetcher nie przekracza granicy strony 4 KiB.** Przy N = 1024 każdy krok to nowa strona,
   więc sprzętowy prefetcher nie pomaga, a każdy dostęp to też chybienie w L1 DTLB.

### Drugi hamulec: łańcuch zależności

`sum += …` to łańcuch: każde dodawanie czeka na wynik poprzedniego. Bez `-ffast-math`
kompilator nie może rozbić sumy na kilka niezależnych, bo zmieniłby wynik (o tym niżej).
Tempo jest więc ograniczone **opóźnieniem** dodawania, a nie przepustowością jednostek.

Opóźnienia zmierzone metodą proporcji: łańcuch `add rejestr, rejestr` (1 cykl) wyznacza
długość cyklu, a łańcuch badanej instrukcji podzielony przez niego daje opóźnienie w cyklach.
Kontrola: `imul` wyszło 3,04 przy oczekiwanych 3.

| | Wynik |
|---|---|
| zegar rdzenia (obecne ustawienia zasilania) | **~2,84 GHz** |
| `vaddss` (dodawanie FP) | **2 cykle** |
| `vfmadd231ss` (FMA) | **4 cykle** |

Pułapka: łańcuch `add rejestr, stała` dał „zegar 15 GHz”. Rdzenie Golden/Raptor Cove wykonują
dodawanie stałej już na etapie przemianowywania rejestrów (*renaming*), więc taki łańcuch nie
trwa 1 cykl na instrukcję. Wzorzec do pomiaru musi być instrukcją, której procesor nie umie
skrócić.

**Granica dla wersji naiwnej:** 1 iteracja pętli k (2 FLOP) na 2 cykle, czyli
1 FLOP/cykl × 2,84 GHz = **2,84 GFLOP/s**. Zmierzone 2,27 to 80% tej granicy. Resztę zjada
prawdopodobnie ręczny gather: 6 instrukcji tasowania na 4 elementy obciąża ten sam port.

**Realny szczyt przy tym zegarze:** 32 FLOP/cykl × 2,84 GHz ≈ **91 GFLOP/s** zamiast 157.
Zegar 2,84 GHz zamiast 4,9 GHz turbo to wynik obecnych ustawień zasilania („Zrównoważony”
i ewentualnie profil producenta laptopa).

### Wyniki (pomiar roboczy, 5 powtórzeń, mediana, 1 wątek)

| N | czas | GFLOP/s | CV | % szczytu (91) | cykli na iterację k |
|---|---|---|---|---|---|
| 256 | 14,8 ms | 2,28 | 5,8% | 2,5% | 2,5 |
| 512 | 181 ms | 1,47 | 8,8% | 1,6% | 3,9 |
| 1000 | 880 ms | 2,27 | 2,9% | 2,5% | 2,5 |
| 1024 | 6213 ms | **0,35** | 5,8% | 0,4% | **16,4** |

Hipoteza przed pomiarem: ok. 2 GFLOP/s dla 256, 512 i 1000 (ogranicza łańcuch) oraz
1–1,5 GFLOP/s dla 1024 (konflikty w L2).

Interpretacja:
- **256 i 1000 mają identyczne tempo**, choć kolumna B przy N = 1000 jest 4× dłuższa. To znak, że
  ogranicza łańcuch dodawań, a nie pamięć. Odczyty B nie zależą od `sum`, więc procesor
  wykonujący instrukcje poza kolejnością (out-of-order) wczytuje je z L2 wiele iteracji naprzód
  i chowa opóźnienie.
- **1024 jest 6,5× wolniejsze niż 1000** przy prawie tej samej pracy i ilości danych. Różni je
  tylko wzorzec adresów. Kolumna potrzebuje 1024 miejsc w L2, a ma 320, więc praktycznie każdy
  odczyt B idzie do L3. 16 cykli na iterację przy opóźnieniu L3 rzędu kilkudziesięciu cykli
  oznacza, że równolegle obsługiwanych jest tylko kilka chybień naraz. Spadek był dużo większy,
  niż przewidywała hipoteza. Dokładny rozkład przyczyn (L3, TLB, liczba chybień w locie)
  wymagałby liczników sprzętowych, a WSL2 ich domyślnie nie udostępnia.
- **512 jest pośrodku (1,47).** Kolumna zajmuje 512 z 640 miejsc w swoich zbiorach L2, czyli 8 z 10
  dróg, a o te same zbiory konkurują inne dane. Część odczytów wypada więc do L3. To
  prawdopodobne wyjaśnienie, ale bez liczników niepotwierdzone.
- CV dla 256, 512 i 1024 przekracza 5%. Różnice 1,5× i 6,5× są dużo większe od szumu, więc
  wnioski się trzymają, ale liczby do README trzeba zmierzyć ponownie w spokojnych warunkach.
- Hipoteza dała dobrą liczbę dla 256 (2,3) z dwóch złych powodów, które się znosiły: opóźnienie
  4 cykle przy 4,5 GHz zamiast 2 cykli przy 2,84 GHz. Zgodność liczby nie dowodzi mechanizmu.

### Poprawność: dlaczego tolerancja i dlaczego względem Σ|a·b|

**Dodawanie floatów nie jest łączne.** Każdy wynik jest zaokrąglany do najbliższej liczby
reprezentowalnej (24 bity mantysy). Przykład: w okolicy 10⁸ odstęp między kolejnymi floatami
to 8, więc:

```
(1e8 + 1) - 1e8 = 0        // 1e8 + 1 rounds back to 1e8
(1e8 - 1e8) + 1 = 1
```

Inna kolejność sumowania (inny porządek pętli, bloki w tilingu, podział na wątki, inne użycie FMA)
oznacza inne zaokrąglenia, a więc wynik różny na ostatnich bitach. Test nie może wymagać
równości bit w bit.

**Oszacowanie błędu iloczynu skalarnego długości n:**

```
|fl(Σ a_k·b_k) − Σ a_k·b_k|  ≤  ~n·u · Σ|a_k·b_k|        u = 2⁻²⁴ ≈ 6·10⁻⁸
```

Test sprawdza każdy element: `|C_ij − ref_ij| ≤ n·eps · Σ_k |a_ik·b_kj|`, gdzie eps = 2u
(2× zapasu), a referencja jest liczona w `double`.

Dlaczego względem Σ|a·b|, a nie względem wyniku: gdy składniki się znoszą, wynik może być
bliski 0, a błąd zaokrągleń nie znika. Test typu `|C − ref| / |ref|` dawałby wtedy fałszywe
błędy dla losowych danych.

Zmierzony błąd jest dużo mniejszy niż oszacowanie najgorszego przypadku, bo błędy zaokrągleń
mają losowe znaki i częściowo się znoszą:

| n | max \|błąd\| / Σ\|a·b\| | granica n·eps |
|---|---|---|
| 1 | 2,0·10⁻⁸ | 1,2·10⁻⁷ |
| 64 | 1,6·10⁻⁷ | 7,6·10⁻⁶ |
| 256 | 2,1·10⁻⁷ | 3,1·10⁻⁵ |

**Testy i co każdy łapie:**
- `HandComputed2x2`: ręcznie policzony wynik. Przy okazji łapie pomylenie A·B z B·A.
- `IdentityIsExact`: A·I = A bit w bit, bo mnożenie przez 1 i dodawanie 0 nie zaokrągla.
- `OverwritesC`: C wypełnione NaN przed wywołaniem. Łapie kernel, który dodaje do starej
  zawartości C zamiast ją nadpisać. Ważne od etapu 2, gdzie i-k-j akumuluje w C.
- `NaiveVsExact`: rozmiary 1, 2, 7, 16, 64, 127, 129, 256. Nieparzyste rozmiary przydadzą się
  przy tilingu, gdzie ostatni blok jest mniejszy od pozostałych.
- `Tolerance.CatchesMissingTerm`: test testu. Usuwamy jeden składnik sumy i sprawdzamy, że
  porównanie to wykrywa. Tolerancja (~5·10⁻⁴) jest ok. 2000× mniejsza od pojedynczego składnika.
- A i B mają różne ziarna (42 i 43). Przy A = B kernel liczący B·A zamiast A·B przeszedłby testy.

### saxpy po wyrównaniu buforów do 64 B

L1: mediana 33,8 GFLOP/s, CV 4,6%, zakres 30,8–35,4. Wolnej grupy 22–25 GFLOP/s już nie ma.
To potwierdza diagnozę z etapu 0: przyczyną były zapisy przecinające granicę linii cache.
