# SF10: obligacje oszczędnościowe Skarbu Państwa

Sub-fundusz SF10 obejmuje detaliczne obligacje oszczędnościowe Skarbu Państwa. Obligacje te nie są
notowane, więc ich stopy zwrotu wynikają z warunków emisji. Kod: `pf.data.savings_bonds`, konfiguracja:
`config/savings_bonds.yaml`, `config/subfunds.yaml` (`SF10`, `deterministic: true`).

## Dane

Zbiór „Obligacje oszczędnościowe Skarbu Państwa: emisje i oprocentowanie” (Błażewski 2026, obrotomierz.pl,
[10.5281/zenodo.23208023](https://doi.org/10.5281/zenodo.23208023), licencja CC BY 4.0), oparty na danych
Ministerstwa Finansów (dane.gov.pl) i listach emisyjnych. Pliki (kopia w `data/raw/savings_bonds/`):

- `emisje.csv`: seria, rodzina, okno sprzedaży, termin wykupu, okres odsetkowy, kapitalizacja, stopa
  pierwszego okresu, opłata za wcześniejszy wykup, cena zamiany;
- `oprocentowanie.csv`: stopa każdej serii w każdym okresie odsetkowym rozpoczętym do 2026-09-30.

## Tickery

| ticker | rodzina | termin | oprocentowanie | odsetki | sprzedaż w danych |
|---|---|---|---|---|---|
| OTS.SP | OTS | 3 mies. | stałe | przy wykupie | od 2017-10 |
| DOS.SP | DOS | 2 lata | stałe | kapitalizacja roczna | do 2022-05 |
| DOR.SP | DOR | 2 lata | stopa referencyjna NBP + marża | kupon miesięczny | od 2022-06 |
| ROR.SP | ROR | 1 rok | stopa referencyjna NBP + marża | kupon miesięczny | od 2022-06 |
| TOZ.SP | TOZ | 3 lata | WIBOR 6M + marża | kupon półroczny | 2012-05 – 2022-07 |
| TOS.SP | TOS | 3 lata | stałe | kapitalizacja roczna | od 2022-08 |
| COI.SP | COI | 4 lata | inflacja + marża (od 2. roku) | kupon roczny | cały okres |
| EDO.SP | EDO | 10 lat | inflacja + marża (od 2. roku) | kapitalizacja roczna | cały okres |

Pominięto rodziny ROS i ROD, sprzedawane wyłącznie beneficjentom programu „Rodzina 800+”.

## Indeks wartości

Indeks rodziny F to wartość pozycji kupionej w pierwszej serii F w sprzedaży od 2005-01-01 i rolowanej
przy wykupie (100 = wartość w dniu pierwszego zakupu, PLN).

Seria kupiona w dniu p ma okresy odsetkowe [b_{k−1}, b_k), b_k = p + k·m miesięcy, gdzie m = 1, 6, 12
albo cały termin (OTS). Stopa okresu k to r_k (ułamek p.a.), a jego długość w latach τ_k = m/12.
Wartość jednej obligacji o nominale 100 zł w dniu d ∈ [b_{k−1}, b_k) wynosi

  u(d) = 100 · Π_{j<k} (1 + r_j τ_j) · (1 + r_k τ_k · (d − b_{k−1}) / (b_k − b_{k−1})).

Odsetki narastają więc liniowo w okresie. Na jego końcu są kapitalizowane (EDO, DOS, TOS) albo wypłacane
jako kupon (COI, TOZ, ROR, DOR), który w indeksie całkowitego zwrotu jest reinwestowany w tę samą pozycję.

W dniu wykupu b_K pozycja jest rolowana w serię tej rodziny w sprzedaży. Zamiana odbywa się 1:1 na sztuki
po cenie zamiany c (zwykle 99,90 zł; brak danych – 100 zł). Dyskonto 100 − c przysługuje więc tylko
wykupywanym obligacjom, a wypłacone odsetki są reinwestowane w zwykłym zakupie po 100 zł:

  V⁺ = V + n · (100 − c),

gdzie n to liczba wykupywanych obligacji. Przy kapitalizacji (EDO, DOS, TOS) i OTS n to liczba obligacji
kupionych w poprzednim rolowaniu, bo skapitalizowane odsetki nie podlegają zamianie. Przy kuponie
(COI, TOZ, ROR, DOR) n obejmuje też obligacje dokupione z reinwestowanych kuponów,
n = jednostki · Π_{j<K}(1 + r_j τ_j), bez kuponu ostatniego okresu wypłacanego razem z wykupem.
Ułamkowe części obligacji pominięto. Jeśli sprzedaż serii
zaczyna się kilka dni po wykupie (≤ 10 dni), środki do tego dnia nie są oprocentowane. Brak serii
w sprzedaży kończy indeks (DOS.SP: 2023-01-03, TOZ.SP: 2024-05-01).

Stopy okresów rozpoczętych po dacie danych MF przyjęto równe stopie poprzedniego okresu. Podatek od
dochodów kapitałowych pominięto, tak jak dla pozostałych aktywów.

Wynik: `data/raw/savings_bonds/index.csv` (kalendarz dzienny) i `holdings.csv` (posiadane serie).
Dzienna log-stopa na sesji t panelu danych to r_t = ln V(t) − ln V(t′), gdzie t′ jest poprzednią sesją,
więc stopa sesji obejmuje wszystkie dni kalendarzowe od t′ (`data/processed/log_returns_pln.csv`).

## Opłata za wcześniejszy wykup

Sprzedaż obligacji przed terminem wykupu obciąża opłata z listu emisyjnego o [zł/obligację]. Jako ułamek
wartości obligacji:

  f(d) = min(o, a(d)) / (100 + a(d)),

gdzie a(d) to odsetki narosłe od zakupu (kapitalizacja) albo od początku bieżącego okresu (kupon).
Dla OTS wcześniejszy wykup następuje bez odsetek, więc f = a / (100 + a). Ograniczenie opłaty do narosłych
odsetek odpowiada listom emisyjnym obecnych serii i przyjęto je dla wszystkich serii. Brak opłaty w danych
(część serii sprzed 2016 r.) oznacza f = 0.

W backteście (`pf.backtest.realized.realize`) koszt w dniu rebalansowania t wynosi

  k_t = c · Σ_i |x_{t,i} − x̃_{t,i}| + Σ_i f_i(t) · (x̃_{t,i} − x_{t,i})⁺,

c = 0,1%, x̃ – wagi po dryfie. Wykup w terminie i rolowanie nie są obciążone opłatą.
Plik: `data/processed/savings_bonds_redemption_cost.csv` (`config/backtest.yaml`: `redemption_cost`).

## Margines deterministyczny

Stopa obligacji w horyzoncie H = 21 sesji jest znana ex ante (oprocentowanie bieżącego okresu), z wyjątkiem
zmiany stopy na początku nowego okresu odsetkowego. Rozkład brzegowy jest więc zdegenerowany (punktowy).
Model GJR-GARCH nie ma tu zastosowania, a kopuła rozkładu niecągłego nie jest jednoznaczna (tw. Sklara).
Dlatego (`pf.models.margins.fit_deterministic`):

- prognoza dziennej log-stopy to μ̂ = (1/n) Σ_{t=T−n+1}^{T} r_t, n = 21 (`margins.deterministic_sessions`);
- w scenariuszach r_h = μ̂ dla każdego h, więc R_i = exp(H·μ̂) − 1 we wszystkich scenariuszach;
- aktywa SF10 nie wchodzą do panelu reszt, modeli zależności, testu Vuonga ani log-score'u; kopuła
  generuje wartości tylko dla aktywów stochastycznych, a wspólne liczby losowe pozostają bez zmian;
- dopuszczenie do okna wymaga 21 obserwacji (`rebalance.min_obs_deterministic`) i notowania w sesji t−1
  (wygasły indeks nie jest kupowany).

Konsekwencje w optymalizacji:

- **CE.** W SF10 CE_γ(w) = Σ_i w_i R_i, więc poziom 1 wybiera rodziny o najwyższej znanej stopie
  w granicach c_SF. Na poziomie 2 SF10 jest składnikiem o zerowej wariancji.
- **Kara za zależność.** Pseudo-obserwacje stałej kolumny są wiązane (V = 1/2), więc Λ_ij(q) = 0
  dla q < 1/2. W ρ_S korelacje stałej kolumny przyjęto jako 0. SF10 nie zwiększa więc kary D(W).
- **ES.** Strata SF10 jest stała (−R), a SE(ÊS) portfela kontrolnego SF10 wynosi 0.
- **`mv_hist`.** Średnie i macierz kowariancji (Ledoit–Wolf) z historycznych stóp okna, jak dla innych SF.
- **MC ex post.** FHS nie jest liczony dla szeregów SF10, a w wielowymiarowym MC SF10 ma stałą stopę
  exp(H·μ̂) − 1 (średnia z próby) i nie wchodzi do R-vine'a.

## Ograniczenia

- **Rocznik serii.** Indeks odzwierciedla pozycję kupioną w 2005 r. (lub przy debiucie rodziny) i rolowaną.
  Inwestor zwiększający udział SF10 w dniu t kupuje serię bieżącą, o innej marży i stopie pierwszego okresu.
- **Rolowanie przy wykupie.** Przyjęto rolowanie w tę samą rodzinę po cenie zamiany. W rzeczywistości zamiana
  jest dobrowolna, a jej warunki określa list emisyjny.
- **Prognoza stopy.** μ̂ z ostatnich 21 sesji nie uwzględnia znanej z góry zmiany stopy w nowym okresie
  odsetkowym (np. rocznica EDO/COI z nową inflacją). Błąd prognozy jest zawarty w zrealizowanych stopach.
- **Opłata.** Ograniczenie opłaty do narosłych odsetek przyjęto dla wszystkich serii.
- **Podatek** od dochodów kapitałowych (19%) pominięto, tak jak dla akcji i ETF-ów.
