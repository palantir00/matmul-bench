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
- Przed i po każdej serii pomiarów uruchamiamy `./build/clock_probe` (od etapu 2), żeby wiedzieć,
  przy jakim zegarze powstały wyniki, i wychwycić przegrzanie.
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

---

## Etap 2: zamiana kolejności pętli (i-k-j)

### Kod

```cpp
for i: { C[i][:] = 0; for k: { a = A[i][k]; for j: C[i][j] += a * B[k][j]; } }
```

Te same mnożenia i dodawania, w tej samej liczbie. Zmienia się tylko kolejność, w jakiej
odwiedzamy elementy. Pętla wewnętrzna to teraz dokładnie saxpy z etapu 0:
**wiersz C += a × wiersz B**.

### Co czyta jedna iteracja pętli wewnętrznej

```
i-j-k (naive): jeden element C = wiersz A · kolumna B

          A                    B                    C
     k ───────►           j                    j
  i [■■■■■■■■■■]     k  [ ■        ]     i  [ ■        ]
                     │  [ ■        ]
                     │  [ ■        ]       każdy ■ w B to inna linia cache,
                     ▼  [ ■        ]       skok o n·4 B (przy N=1024 to 4 KiB,
                        [ ■        ]       czyli nowa strona pamięci)

i-k-j: cały wiersz C += jedna liczba z A × cały wiersz B

          A                    B                    C
     k                         j ───────►           j ───────►
  i [ ■        ]     k  [■■■■■■■■■■]     i  [■■■■■■■■■■]
    jedna liczba,        ciągły wiersz,       ciągły wiersz,
    ta sama dla          czytany od           ten sam dla wszystkich k,
    całego wiersza       początku do końca    siedzi w L1 (≤ 4 KiB)
```

Dla ustalonego `i` pętla po `k` przechodzi przez B wiersz po wierszu, czyli przez **całą
macierz B po kolei**, tak jak leży w pamięci:

```
pamięć B:  [wiersz 0][wiersz 1][wiersz 2] ... [wiersz n-1]   ──► jeden ciągły strumień, raz na każde i
```

### Dlaczego to pomaga

1. **Linie cache wykorzystane w całości.** Z każdej wczytanej 64-bajtowej linii B używamy
   wszystkich 16 floatów, zamiast jednego.
2. **Prefetcher ma łatwe zadanie.** Widzi jeden rosnący strumień adresów i ściąga dane
   z wyprzedzeniem. Prefetcher nie przekracza granicy strony 4 KiB, więc mniej więcej co 4 KiB
   (przy N=1024 co jeden wiersz B) musi zacząć strumień od nowa. To niewielki koszt.
3. **Nie ma już łańcucha zależności.** W pętli j każde `C[i][j]` jest niezależne od sąsiednich,
   więc procesor może liczyć wiele naraz. Element `C[i][j]` dalej dostaje składniki w kolejności
   k = 0, 1, …, n−1, tak jak w wersji naiwnej, więc kompilator wektoryzuje bez `-ffast-math`.
   Nie przestawia sumy, tylko liczy 8 różnych sum jednocześnie.
4. **Nie ma konfliktów zbiorów przy potęgach dwójki.** Konflikty z etapu 1 wynikały ze skoku
   o n·4 B w dół kolumny. Tutaj adresy idą po kolei, więc N=1024 nie powinno się już niczym
   wyróżniać.

### Co zrobił kompilator

Raport wektoryzacji GCC (`-fopt-info-vec`):

```bash
g++ -O3 -std=c++17 -march=native -Iinclude -fopt-info-vec-optimized -c src/ikj.cpp -o /dev/null
g++ -O3 -std=c++17 -march=native -Iinclude -fopt-info-vec-missed    -c src/naive.cpp -o /dev/null
```

```
ikj.cpp:17:31: optimized: loop vectorized using 32 byte vectors
ikj.cpp:17:31: optimized:  loop versioned for vectorization because of possible aliasing
ikj.cpp:17:31: optimized: loop vectorized using 16 byte vectors
naive.cpp:13:50: missed: not vectorized: not suitable for strided load
```

- **32 byte vectors**: pętla j liczy 8 floatów naraz (rejestr AVX2 `ymm`).
- **16 byte vectors**: osobna, krótsza wersja na resztę, gdy `n` nie dzieli się przez 8.
- **versioned because of possible aliasing**: kompilator nie wie, że wiersz C nie nakłada się
  w pamięci na wiersz B (wskaźniki tego typu mogą wskazywać na ten sam obszar). Dlatego
  generuje dwie wersje pętli i przed każdym jej uruchomieniem sprawdza adresy. Gdy obszary się
  nie nakładają, a u nas tak jest zawsze, idzie wersja wektorowa. Sprawdzenie odbywa się raz na
  wiersz (n² razy przy n³ pracy), więc kosztuje niewiele. Słowo `__restrict` by je usunęło, ale
  to rozszerzenie kompilatora, a nie standard C++.
- **naive: strided load**: kolumny B nie da się wczytać jednym odczytem wektorowym.
- Zerowanie wiersza C kompilator zamienił na wywołanie `memset`.

Gorąca pętla w asemblerze (`objdump -d build/CMakeFiles/matmul.dir/src/ikj.cpp.o`):

```asm
vbroadcastss %xmm1,%ymm2              ; a = A[i][k] skopiowane do 8 pozycji rejestru
loop:
vmovups      (%rax,%rdx),%ymm0        ; 8 floatów z wiersza B
vfmadd213ps  (%rcx,%rdx),%ymm2,%ymm0  ; ymm0 = a * B + C (8 floatów C wczytane w tej samej instrukcji)
vmovups      %ymm0,(%rcx,%rdx)        ; zapis 8 floatów C
add          $0x20,%rdx               ; następne 32 B
cmp / jne    loop
```

Na każdą iterację przypada 1 FMA (16 FLOP), 2 odczyty i 1 zapis. Pętla nie jest rozwinięta
(GCC tego domyślnie nie robi), ale iteracje są od siebie niezależne, więc procesor nakłada
kolejne na siebie i opóźnienie FMA (4 cykle) nie ogranicza tempa.

### Wyniki (pomiar roboczy, 5 powtórzeń, mediana, 1 wątek)

Warunki: zasilacz, tryb „Zrównoważony”, `clock_probe` przed i po: **~3,0 GHz** (2,85–3,10),
bez spadku pod koniec. Szczyt przy tym zegarze: 32 × 3,0 ≈ 96 GFLOP/s. Naive i i-k-j zmierzone
w jednym przebiegu.

| N | naive | i-k-j | CV i-k-j | przyspieszenie | FLOP/cykl | % szczytu | hipoteza |
|---|---|---|---|---|---|---|---|
| 256 | 2,65 | **36,0** | 2,3% | **13,6×** | 12,0 | 38% | 20–30 |
| 512 | 1,72 | **30,9** | 4,2% | **18×** | 10,3 | 32% | 15–25 |
| 1000 | 2,68 | **18,0** | 8,4% | **6,7×** | 6,0 | 19% | 10–20 |
| 1024 | 0,57 | **20,8** | 2,3% | **36×** | 6,9 | 22% | 10–20 |

### Porównanie z hipotezą

- **Jakościowo wszystko się zgadza.** Pętla jest zwektoryzowana (8 floatów, FMA), przepaść
  między 1000 a 1024 zniknęła (18,0 i 20,8, różnica w granicach rozrzutu przy CV 8%),
  przyspieszenie jest największe tam, gdzie wersja naiwna cierpiała najbardziej (1024: 36×).
- **Dla 256 i 512 hipoteza była zbyt ostrożna.** Założenie: dostarczanie B z L2 będzie
  hamulcem, bo saxpy w L2 dawał tylko ~17 GFLOP/s. Błąd w rozumowaniu: saxpy w L2 przepycha
  przez L2 12 B na element (x, y i zapis y), a i-k-j tylko 4 B (sam wiersz B, bo wiersz C siedzi
  w L1). Wynik dla 256 (36,0) to tyle, co saxpy w całości w L1 (33,8). **Przy N=256 położenie
  danych przestało być ograniczeniem, ogranicza sama pętla.**
- **Dla 1000/1024 lepiej sprawdziłby się prosty model niż szeroki przedział** (opisany niżej).

### Model: przepustowość poziomu, na którym leży B

Na każde FMA (2 FLOP) pętla wczytuje jeden float B (4 B) z poziomu, na którym leży B, czyli
**0,5 FLOP na bajt** z tego poziomu. Wiersz C jest w L1 i prawie nic nie kosztuje. Górna
granica to więc 0,5 × (przepustowość tego poziomu), ale nie więcej niż pozwala sama pętla.

| N | B (rozmiar) | leży w | przepustowość (z saxpy, etap 0) | 0,5 × przepustowość | zmierzone |
|---|---|---|---|---|---|
| 256 | 256 KiB | L2 | ~106 GB/s | ~53 | 36,0, ogranicza pętla (≈ saxpy w L1) |
| 1024 | 4 MiB | L3 | ~49 GB/s | ~24 | 20,8 (86% modelu) |

Przy N=1024 i-k-j wyciąga z L3 ok. 20,8 / 2 × 4 B ≈ 42 GB/s, czyli prawie tyle, ile jeden rdzeń
może pobrać z L3. To pierwsze zetknięcie z modelem **roofline**: osiągalne GFLOP/s =
min(możliwości obliczeniowe, AI × przepustowość pamięci). Rozwiniemy go w etapie 5.

Wniosek na etap 3: dla N=1024 ograniczeniem jest to, że **cała macierz B przejeżdża z L3
osobno dla każdego wiersza i**. Kafelkowanie (tiling) ma sprawić, że kawałek B wczytany raz
zostanie użyty dla wielu wierszy, zanim wypadnie z cache. Dla N=256 kafelkowanie niewiele da,
bo tam B już siedzi w L2, a ogranicza pętla.

### Dlaczego i-k-j nie dojdzie do szczytu

Na każde FMA przypadają 3 operacje na pamięci: odczyt B, odczyt C i zapis C. Według
dokumentacji mikroarchitektury (nie mierzyliśmy tego) rdzeń P wykonuje na cykl do 2 FMA,
ale tylko ok. 3 odczyty i 2 zapisy wektorowe. Przy 2 odczytach na FMA odczyty pozwalają na
najwyżej 1,5 FMA na cykl, i to w idealnym przypadku. W praktyce ten kształt pętli (saxpy w L1)
osiąga ok. 12 FLOP/cykl na 32 możliwe.

Wiersz C jest wczytywany i zapisywany przy **każdym** k, choć tak naprawdę wystarczyłoby
wczytać go raz i zapisać raz. Rozwiązaniem jest trzymanie fragmentu C w rejestrach przez wiele
kolejnych k (*register blocking*). Tak robią mikrojądra bibliotek BLAS (etap 6). Sama zmiana
położenia danych tego nie naprawi.

### Bit w bit z wersją naiwną: tylko 19–67% elementów

i-k-j dodaje składniki w tej samej kolejności co wersja naiwna, a mimo to większość wyników
różni się na ostatnich bitach:

| n | 1 | 2 | 7 | 16 | 64 | 256 |
|---|---|---|---|---|---|---|
| identyczne z naive | 100% | 100% | 67% | 42% | 27% | 19% |

Przyczyna to FMA. W i-k-j każdy krok to `vfmadd`: `c + a·b` liczone dokładnie i zaokrąglane
**raz**. Główna pętla wersji naiwnej mnoży (`vmulps`) i dodaje (`vaddss`) osobno, czyli
zaokrągla **dwa razy**. Dla n = 1 i 2 wersja naiwna działa wyłącznie w pętli dla reszty
(n mod 4), która też używa FMA, stąd 100%. GCC dla C++ domyślnie łączy `a*b + c` w FMA
(`-ffp-contract=fast`), gdy tylko może.

Wnioski:
- Ta sama kolejność sumowania nie wystarcza do identycznych bitów. Liczą się też instrukcje,
  które wybierze kompilator. Dlatego testy zawsze używają tolerancji.
- FMA nie pogarsza dokładności, raczej ją poprawia (jedno zaokrąglenie zamiast dwóch). Błąd
  względem `double` jest u obu wersji tego samego rzędu (~2·10⁻⁷ dla n=256). Różnica
  i-k-j − naive dla n=256 to najwyżej 9·10⁻⁸ × Σ|a·b|, przy tolerancji 6·10⁻⁵.

### Ponowny pomiar wersji naiwnej: dryf między sesjami

| N | etap 1 (2,84 GHz) | etap 2 (~3,0 GHz) | zmiana |
|---|---|---|---|
| 256 | 2,28 | 2,65 | +16% |
| 512 | 1,47 | 1,72 | +17% |
| 1000 | 2,27 | 2,68 | +18% |
| 1024 | 0,35 | 0,57 | **+63%** |

Zegar wzrósł o ok. 6%, a wyniki o 16–63%. Rozrzut wewnątrz każdej sesji (CV 2–9%, poza 512)
jest dużo mniejszy niż różnica między sesjami. Możliwe przyczyny, których nie da się rozdzielić
bez liczników sprzętowych:
- benchmark mógł działać przy innym zegarze niż w chwili pomiaru `clock_probe`;
- inne obciążenie systemu (teraz load average 0,42, w etapie 1 VS Code był aktywny);
- dla 1024, czyli dla pracy ograniczonej przez L3, także zegar części procesora poza rdzeniami
  (*uncore*: L3 i szyna łącząca rdzenie), którym Windows steruje niezależnie.

**Lekcja: wersje porównujemy tylko w obrębie jednego przebiegu.** Dlatego naive jest mierzone
ponownie razem z każdą nową wersją, a liczby do README powstaną w jednej sesji, w ustalonych
warunkach.

### Testy po przebudowie

Testy są teraz sparametryzowane listą kerneli (`kAllKernels`). Nowa wersja to jedna linijka
i od razu dostaje komplet testów:
- `KernelBasics` (2×2 policzone ręcznie, macierz jednostkowa, NaN w C) dla każdej wersji;
- `KernelVsExact`: błąd względem `double` ≤ n·eps·Σ|a·b|, 8 rozmiarów;
- `KernelVsNaive`: każda wersja poza naive porównana z naive, tolerancja 2·n·eps·Σ|a·b|.
  Skoro obie mieszczą się w n·eps·Σ|a·b| od wartości dokładnej, to od siebie dzieli je najwyżej
  dwa razy tyle (nierówność trójkąta). Naive jest wzorcem, bo jego poprawność widać
  z samego kodu, a dodatkowo sam jest sprawdzany względem `double`.

`OverwritesC` jest teraz naprawdę potrzebny. i-k-j dodaje do C, więc kernel, który zapomniałby
wyzerować wiersz, dałby w wyniku NaN.

### Nowe narzędzie: `clock_probe`

`./build/clock_probe` mierzy zegar rdzenia i opóźnienia `vaddss`/`vfmadd231ss` metodą
proporcji opisaną w etapie 1. Uruchamiamy go przed i po każdej serii pomiarów.

---

## Etap 3: kafelkowanie (tiling)

### Problem z etapu 2

Dla N=1024 i-k-j przy każdym wierszu `i` przechodzi przez **całą** macierz B (4 MiB, czyli
z L3). Każdy element B jest więc wczytywany z L3 n razy, łącznie n³ floatów. Pętla jest
ograniczona przepustowością L3 (~42 GB/s).

### Pomysł: wzór blokowy

Dzielimy macierze na kafelki T×T. Mnożenie macierzy działa tak samo na blokach, jak na liczbach:

```
C_IJ = Σ_K  A_IK · B_KJ            (I, J, K: numery kafelków)
```

```
for ii:                       ─┐
  for jj:                      │ wybór kafelka C(ii,jj)
    for kk:                    │ C(ii,jj) += A(ii,kk) · B(kk,jj)
      for i in kafelek:        ─┤
        for k in kafelek:       │ wewnątrz: to samo i-k-j co w etapie 2,
          for j in kafelek:     │ tylko na kawałku T×T
            C[i][j] += A[i][k] * B[k][j]
```

```
           B                               kafelek B(kk,jj) wczytany raz
     ┌────┬────┬────┐                      jest używany przez T kolejnych
     │    │    │    │                      wierszy i (w i-k-j: przez jeden)
     ├────┼────┼────┤
  kk │    │ ██ │    │  ◄── T×T, mieści się w L1/L2
     ├────┼────┼────┤
     │    │    │    │
     └────┴────┴────┘
             jj
```

**Ruch B z L3** spada z n³ do n³/T floatów. Dla N=1024 i T=128: z 4 GiB do 32 MiB.

**Kolejność sumowania się nie zmienia.** kk rośnie w pętli zewnętrznej, k rośnie wewnątrz
kafelka, więc każde `C[i][j]` dostaje składniki w kolejności k = 0…n−1. Wynik jest identyczny
z i-k-j bit w bit dla każdego T. Sprawdza to test `Tiled.BitIdenticalToIkj`.

| T | kafelek B | kafelki A+B+C | gdzie się mieszczą |
|---|---|---|---|
| 16 | 1 KiB | 3 KiB | L1 |
| 32 | 4 KiB | 12 KiB | L1 |
| 64 | 16 KiB | 48 KiB | L1 na styk (48 KiB) |
| 128 | 64 KiB | 192 KiB | L2 |
| 256 | 256 KiB | 768 KiB | L2 (60% pojemności) |

### Hipoteza przed pomiarem

1. N=1000/1024: z ~20 do 28–36 GFLOP/s (~1,5× względem i-k-j).
2. Sufit ~36 GFLOP/s: pętla wewnętrzna ma ten sam kształt co w i-k-j.
3. T=16 tylko ~10–15 GFLOP/s, bo pętla j ma 2 iteracje wektorowe i narzut ich nie zamortyzuje.
4. Optimum T ≈ 64–128.
5. N=256: kafelkowanie nie pobije i-k-j.
6. N=1024 może tracić względem 1000, bo wiersze kafla leżą co 4 KiB (konflikty zbiorów).
7. Wyniki bit w bit takie jak w i-k-j.

### Metodologia: trzy poprawki w trakcie etapu

- **Czas zegarowy zamiast czasu CPU** (`UseRealTime()`). W pierwszym przebiegu kolumna `Time`
  była o ~5% większa od `CPU`. Google Benchmark domyślnie liczy GFLOP/s z czasu CPU, który nie
  obejmuje chwil, gdy system odebrał wątkowi rdzeń. Uczciwy jest czas zegarowy. W etapie 4 czas
  CPU byłby wręcz błędny, bo przy wielu wątkach to suma czasów wszystkich wątków.
- **Losowe przeplatanie powtórzeń** (`--benchmark_enable_random_interleaving`, teraz domyślnie
  w `run_bench.sh`). Powtórzenia wszystkich wariantów idą w losowej kolejności, więc chwilowe
  spowolnienie maszyny rozkłada się na wszystkie warianty, zamiast zepsuć jeden.
- **Trzy niezależne przebiegi**, a tabela pochodzi z ostatniego (7 powtórzeń, przeplatanie).
  Zegar w tej sesji: ~2,8 GHz, szczyt 32 × 2,8 ≈ 90 GFLOP/s.

### Wyniki (GFLOP/s, mediana, czas zegarowy, 1 wątek)

| N | naive | i-k-j | T=16 | T=32 | T=64 | **T=128** | T=256 |
|---|---|---|---|---|---|---|---|
| 256 | 2,4 | 24,2 | 11,9 | 18,2 | 24,6 | **30,7** | 23,6 |
| 512 | 1,7 | 27,0 | 11,4 | 16,7 | 22,9 | **28,5** | 23,3 |
| 1000 | 2,5 | 19,1 | 9,9 | 16,2 | 24,5 | **26,6** | 22,3 |
| 1024 | 0,44 | 18,7 | 10,3 | 14,2 | 21,2 | **24,4** | 21,0 |

CV w ostatnim przebiegu: 1–6%, wyjątki: i-k-j 1024 10%, naive 512 12%, T=16 przy 1024 10%.

Najlepsze T=128 względem poprzednich wersji:

| N | vs i-k-j | vs naive |
|---|---|---|
| 256 | 1,27× (ale i-k-j jest tu niestabilne, patrz niżej) | 13× |
| 512 | 1,06× | 17× |
| 1000 | **1,39×** | 11× |
| 1024 | **1,30×** | **55×** |

### Porównanie z hipotezą

| | hipoteza | wynik | |
|---|---|---|---|
| 1 | 1000/1024: 28–36 GFLOP/s, ~1,5× | 26,6 / 24,4, czyli 1,3–1,4× | kierunek ✓, liczby za optymistyczne |
| 2 | sufit ~36 | żaden wariant nie przekroczył ~31 | ✓ |
| 3 | T=16: 10–15 | 10–12 | ✓ |
| 4 | optimum 64–128 | 128 dla każdego N, a 64 wyraźnie gorsze | częściowo ✓ |
| 5 | N=256: nie lepiej niż i-k-j | w 2 z 3 przebiegów T=128 wygrało z i-k-j o 27% | ✗ (niestabilność, niżej) |
| 6 | 1024 traci względem 1000 | −13% przy T=64, −8% przy T=128 | ✓, efekt umiarkowany |
| 7 | bit w bit = i-k-j | test przechodzi | ✓ |

### Główna lekcja: kafelkowanie to kompromis, a koszt da się zmierzyć

Dla N=256 kafelkowanie **nie ma czego naprawiać**, bo całe A, B i C mieszczą się w L2.
Spadek przy małym T to więc czysty koszt kafelkowania. To kontrolowany eksperyment.

Koszt bierze się z tego, że pętla j ma długość T, a nie n. Każde jej uruchomienie ma stały
narzut: ustawienie pętli, sprawdzenie aliasingu, broadcast `a`, wyjście z pętli. Czas jednego
uruchomienia wyliczony z pomiarów, t(T) = 2T / GFLOP/s [ns]:

| T | 16 | 32 | 64 | 128 |
|---|---|---|---|---|
| t(T) zmierzone | 2,69 ns | 3,52 ns | 5,20 ns | 8,34 ns |
| model 1,88 + 0,0504·T | 2,69 | 3,49 | 5,11 | 8,33 |

Prosta pasuje z dokładnością do 2%. Przy ~2,8 GHz oznacza to:
- **~5 cykli stałego narzutu** na każde uruchomienie pętli j;
- **~1,1 cyklu na 8 floatów** (jedna iteracja wektorowa) pracy właściwej.

Dla T=16 pętla ma 2 iteracje wektorowe, czyli ~2,3 cyklu pracy na ~5,3 cyklu narzutu, więc
tylko 30% czasu idzie na obliczenia. Gdyby narzutu nie było, pętla dawałaby
2 / 0,0504 ≈ **40 GFLOP/s**. T=128 osiąga 77% tej wartości.

Porównanie N=1000 z N=256 przy tym samym T rozdziela oba efekty:
- **T=64: 24,5 wobec 24,6.** Dla N=1000 przy T=64 problem pamięci jest rozwiązany całkowicie,
  a limitem jest wyłącznie narzut krótkiej pętli.
- **T=128: 26,6 wobec 30,7.** Narzut jest mniejszy, ale pojawia się ~13% kosztu po stronie
  pamięci. Prawdopodobnie to wczytywanie nowych kafelków z L3 na początku każdego mnożenia
  kafelków. Prefetcher widzi krótkie strumienie (wiersz kafla to 512 B), a nie jeden ciągły.
  Bez liczników sprzętowych tego nie potwierdzimy.
- **1024 vs 1000** (−8% do −13%): wiersze kafla co 4 KiB wpadają do tych samych zbiorów cache,
  ten sam mechanizm co kolumna w etapie 1, tylko łagodniejszy.

Wnioski:
- Reguła „wybierz największy kafelek, który mieści się w L1” tutaj zawodzi. T=64 mieści się
  w L1, a przegrywa z T=128, bo **w tej konstrukcji T to także długość pętli wewnętrznej**.
- Lepsza konstrukcja rozdziela te dwie rzeczy: kafelki prostokątne (długie w j, krótkie w k
  i i) albo osobne rozmiary bloków dla każdego poziomu cache. Tak robią biblioteki BLAS
  (bloki m_c, k_c, n_c w algorytmie Goto).
- Sufit ~40 GFLOP/s wynika z kształtu pętli (2 odczyty + 1 zapis na FMA). Żaden rozmiar kafla go
  nie podniesie. Potrzebne jest trzymanie fragmentu C w rejestrach (mikrojądro, etap 6).

### Niestabilność wariantów z pętlą długości 256

Wartości z trzech przebiegów, przeliczone na czas zegarowy:

| wariant | przebieg 1 | przebieg 2 | przebieg 3 |
|---|---|---|---|
| i-k-j, N=256 | 31,3 | 23,7 | 24,2 |
| T=256, N=256 | 30,8 | 22,4 | 23,6 |
| T=256, N=1000 | 27,1 | 22,9 | 22,3 |
| T=128, N=256 | 30,4 | 30,2 | 30,7 |
| T=64, N=1000 | 24,8 | 24,7 | 24,5 |

Wszystkie warianty z T ≤ 128 zgadzają się między przebiegami w granicach ~5%. Warianty, w których
pętla j ma 256 elementów, wahają się od ~22 do ~36 GFLOP/s (36 dał i-k-j w etapie 2 i w osobnym
programie testowym). Spadek zegara albo przeniesienie wątku na rdzeń E spowolniłyby wszystkie
warianty, a nie tylko te.

Sprawdzone hipotezy:
1. *4K aliasing między wierszem C a wierszem B.* Program testowy przesuwał C względem B co
   256 B (16 przesunięć, 2 przebiegi). Brak systematycznej zależności, wahania ±3 to szum.
   **Odrzucona.**
2. *Kolorowanie stron.* L2 wybiera zbiór na podstawie fizycznego adresu, a strony 4 KiB system
   przydziela w przypadkowych miejscach, inaczej w każdym procesie. Test: 5 procesów na stronach
   4 KiB i 5 na stronach 2 MiB (fizycznie ciągłych). Na stronach 4 KiB tym razem wszystkie procesy
   dały ~36. Na stronach 2 MiB wynik nie stał się stabilniejszy, a T=128 spadło do ~27.
   Prawdopodobnie przez sam układ eksperymentu: A, B i C leżały co 1 MiB w fizycznie ciągłej
   pamięci, więc trafiały do tych samych zbiorów L2. **Nierozstrzygnięta.**

Przyczyna pozostaje otwarta. Do jej znalezienia potrzebne są liczniki sprzętowe (`perf` na
natywnym Linuksie: chybienia L1/L2, blokady przy odczytach). Wniosek z etapu 2, że i-k-j dla
N=256 ogranicza wyłącznie pętla, trzeba osłabić, bo zdarzają się sesje z wynikiem ~24.
Praktyczny wniosek: T=128 daje podobny najlepszy wynik i jest **przewidywalne**, a to też zaleta.
