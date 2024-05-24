# HoVerNet output format
Ja używałem HoVerNeta z repo autorów zawsze i oni mają cały pipeline zaimplementowany, który zwraca dla każdego obrazka 
trzy pliki:
 - overlay - obrazek z naniesionymi granicami komórek/jąder 
 - mat - scipy mat file, w którym mamy słownik z:
   - inst_map - numpy array w kształcie obrazka w którym każdy piksel ma przypisany numer komórki do której należy
   - inst_uid - lista w której są kolejne id komórek, (id zaczynają się od 1)
   - inst_type - to jest tylko przy klasyfikacji więc nas nie obchodzi
 - json - plik jsonowy który pozwala załadować wyniki jako adnotacje do prgramów do oglądania np. QuPath, komórki 
opisane jako poligony z informacjami o klasyfikacji.

Jak się puszcza tego z monai co jest np. [w tym tutorialu](https://github.com/Project-MONAI/tutorials/tree/main/pathology/hovernet) 
to jeśli dobrze rozumiem to model powinien zwracać słownik w którym są:
 - predykcje tego które piksele są w komórkach
 - predykcje tego jak daleko dany piksel jest od środka najbliższej komórki
 - pedykcje typu (nas nie dotyczy)

Ale jest też zaimplementowane `SlidingWindowHoVerNetInferer` (jest w: `monai.apps.pathology.inferers`) które robi z tego outputu taki jakbyśmy chcieli, 
tj. predykcje instancji komórek i do tego to co wyżej
