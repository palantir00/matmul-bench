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
