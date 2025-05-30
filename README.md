# Паралельна обробка файлів

Обробка файлів вимагає насамперед їх читання, і тому робота з цією темою вимагає особливої уваги.
Цілком можливо, що читання файлів виявиться значно повільнішим за наші обчислення.
В таких випадках паралельне виконання не прискорюватиме виконання задачі.
Отже, важливо, щоб операції які ми виконували над файлами вимагали більше часу ніж їх читання.
На щастя, існує ціла категорія алгоритмів які працюють в тому числі з файлами і часто вимагають багато часу на виконання.
Йде мова про алгоритми хешування.
У рамках цієї лабораторної роботи використовується алгоритм хешування `SHA-512` з бібліотеки `OpenSSL`.

# Ідея програми

1. Через аргументи командного рядка отримуємо список файлів та папок які необхідно опрацювати.
2. Рекурсивно шукаємо усі файли в зазначених директоріях.
3. Сортуємо файли за розміром та рівномірно розподіляємо їх між процесами.
4. Кожний процес послідовно зчитує призначені йому файли і виконує їх хешування.
5. Кореневий процес збирає усі хеші й виводить їх на екран у відповідності зі знайденими файлами.

# MPI
> `#define MPI`

## Виконання програми

```PowerShell
mpiexec.exe -n <кількість процесів> <шлях до виконуваного файлу> <шляхи до файлів та папок до хешування>
```

Для заміру часу можна використати наступний синтаксис:

```PowerShell
Measure-Command {mpiexec.exe -n <n> <exe> <paths>}
```

## Тестування

Для автоматизованого заміру часу створено наступний скрипт, `mpi.ps1`:

```PowerShell
param (
    [string]$exe,
    [int]$min = 2,
    [int]$max = 12,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$rest
)

if (-not (Test-Path $exe)) {
    Write-Error "Executable not found: $exe"
    exit 1
}

Write-Host ("n" + " | " + "Ticks" + " | " + "Seconds")
Write-Host ("--" + "|" + "-------" + "|" + "--------")

for ($n = $min; $n -le $max; $n++) {
    $duration = Measure-Command {
        mpiexec.exe -n $n $exe @rest
    }

    $line = "$n" + " | " + $duration.Ticks + " | " + $duration.Seconds
    Write-Host $line
}

```

Ось приклад використання цього скрипту для заміру часу роботи `Debug` та `Release` конфігурації, `mpi-session.txt`:

```PowerShell
**********************
Windows PowerShell transcript start
Start time: 20250423215036
Username: FYODOR-PC\Fyodor
RunAs User: FYODOR-PC\Fyodor
Configuration Name: 
Machine: FYODOR-PC (Microsoft Windows NT 10.0.17763.0)
Host Application: C:\Windows\system32\WindowsPowerShell\v1.0\PowerShell.exe
Process ID: 13944
PSVersion: 5.1.17763.1971
PSEdition: Desktop
PSCompatibleVersions: 1.0, 2.0, 3.0, 4.0, 5.0, 5.1.17763.1971
BuildVersion: 10.0.17763.1971
CLRVersion: 4.0.30319.42000
WSManStackVersion: 3.0
PSRemotingProtocolVersion: 2.3
SerializationVersion: 1.1.0.1
**********************
Transcript started, output file is mpi-session.txt
PS C:\Users\Fyodor\Documents\3 курс\Семестр 2\РПП\HashFiles> .\mpi.ps1 -exe .\bin\Debug\MPI.exe -min 1 -max 12 "C:\Users\Fyodor\The Legend of Vox Machina"
n | Ticks | Seconds
--|-------|--------
1 | 280782226 | 28
2 | 135149429 | 13
3 | 101517758 | 10
4 | 82753185 | 8
5 | 73259645 | 7
6 | 101545955 | 10
7 | 87288432 | 8
8 | 94367712 | 9
9 | 90972665 | 9
10 | 91759717 | 9
11 | 93962135 | 9
12 | 92083101 | 9
PS C:\Users\Fyodor\Documents\3 курс\Семестр 2\РПП\HashFiles> .\mpi.ps1 -exe .\bin\Release\MPI.exe -min 1 -max 12 "C:\Users\Fyodor\The Legend of Vox Machina"
n | Ticks | Seconds
--|-------|--------
1 | 247645292 | 24
2 | 131467868 | 13
3 | 95841889 | 9
4 | 74021766 | 7
5 | 82841872 | 8
6 | 94622873 | 9
7 | 79528880 | 7
8 | 87649260 | 8
9 | 85320939 | 8
10 | 86951678 | 8
11 | 88959910 | 8
12 | 87337689 | 8
PS C:\Users\Fyodor\Documents\3 курс\Семестр 2\РПП\HashFiles> Stop-Transcript
**********************
Windows PowerShell transcript end
End time: 20250423215525
**********************
```

## Порівняння результатів

### `Debug`

n | Ticks | Seconds
--|-------|--------
 1 | 280782226 | 28
 2 | 135149429 | 13
 3 | 101517758 | 10
 4 |  82753185 |  8
 5 |  73259645 |  7
 6 | 101545955 | 10
 7 |  87288432 |  8
 8 |  94367712 |  9
 9 |  90972665 |  9
10 |  91759717 |  9
11 |  93962135 |  9
12 |  92083101 |  9

### `Release`

n | Ticks | Seconds
--|-------|--------
 1 | 247645292 | 24
 2 | 131467868 | 13
 3 |  95841889 |  9
 4 |  74021766 |  7
 5 |  82841872 |  8
 6 |  94622873 |  9
 7 |  79528880 |  7
 8 |  87649260 |  8
 9 |  85320939 |  8
10 |  86951678 |  8
11 |  88959910 |  8
12 |  87337689 |  8

# OpenMP
> `// #define MPI`

## Виконання програми

```PowerShell
OpenMP.exe <кількість процесів> <шляхи до файлів та папок до хешування>
```

Для заміру часу можна використати наступний синтаксис:

```PowerShell
Measure-Command {OpenMP.exe <n> <paths>}
```

## Тестування

Для автоматизованого заміру часу створено наступний скрипт, `omp.ps1`:

```PowerShell
param (
    [string]$exe,
    [int]$min = 2,
    [int]$max = 12,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$rest
)

if (-not (Test-Path $exe)) {
    Write-Error "Executable not found: $exe"
    exit 1
}

Write-Host ("n" + " | " + "Ticks" + " | " + "Seconds")
Write-Host ("--" + "|" + "-------" + "|" + "--------")

for ($n = $min; $n -le $max; $n++) {
    $duration = Measure-Command {
        & $exe $n @rest
    }

    $line = "$n" + " | " + $duration.Ticks + " | " + $duration.Seconds
    Write-Host $line
}
```

Ось приклад використання цього скрипту для заміру часу роботи `Debug` та `Release` конфігурації, `omp-session.txt`:

```PowerShell
**********************
Windows PowerShell transcript start
Start time: 20250423215548
Username: FYODOR-PC\Fyodor
RunAs User: FYODOR-PC\Fyodor
Configuration Name: 
Machine: FYODOR-PC (Microsoft Windows NT 10.0.17763.0)
Host Application: C:\Windows\system32\WindowsPowerShell\v1.0\PowerShell.exe
Process ID: 13944
PSVersion: 5.1.17763.1971
PSEdition: Desktop
PSCompatibleVersions: 1.0, 2.0, 3.0, 4.0, 5.0, 5.1.17763.1971
BuildVersion: 10.0.17763.1971
CLRVersion: 4.0.30319.42000
WSManStackVersion: 3.0
PSRemotingProtocolVersion: 2.3
SerializationVersion: 1.1.0.1
**********************
Transcript started, output file is omp-session.txt
PS C:\Users\Fyodor\Documents\3 курс\Семестр 2\РПП\HashFiles> .\omp.ps1 -exe .\bin\Debug\OMP.exe -min 1 -max 12 "C:\Users\Fyodor\The Legend of Vox Machina"
n | Ticks | Seconds
--|-------|--------
1 | 237696213 | 23
2 | 142994144 | 14
3 | 118240891 | 11
4 | 111594518 | 11
5 | 111090416 | 11
6 | 107153362 | 10
7 | 112369270 | 11
8 | 107902148 | 10
9 | 108687417 | 10
10 | 102309184 | 10
11 | 105890221 | 10
12 | 103824777 | 10
PS C:\Users\Fyodor\Documents\3 курс\Семестр 2\РПП\HashFiles> .\omp.ps1 -exe .\bin\Release\OMP.exe -min 1 -max 12 "C:\Users\Fyodor\The Legend of Vox Machina"
n | Ticks | Seconds
--|-------|--------
1 | 233479887 | 23
2 | 132001706 | 13
3 | 98215813 | 9
4 | 105532869 | 10
5 | 95057497 | 9
6 | 87390268 | 8
7 | 89011266 | 8
8 | 91075460 | 9
9 | 90258810 | 9
10 | 89722936 | 8
11 | 91529955 | 9
12 | 91633545 | 9
PS C:\Users\Fyodor\Documents\3 курс\Семестр 2\РПП\HashFiles> Stop-Transcript
**********************
Windows PowerShell transcript end
End time: 20250423220212
**********************
```

## Порівняння результатів

### Debug

n | Ticks | Seconds
--|-------|--------
 1 | 237696213 | 23
 2 | 142994144 | 14
 3 | 118240891 | 11
 4 | 111594518 | 11
 5 | 111090416 | 11
 6 | 107153362 | 10
 7 | 112369270 | 11
 8 | 107902148 | 10
 9 | 108687417 | 10
10 | 102309184 | 10
11 | 105890221 | 10
12 | 103824777 | 10

### Release

n | Ticks | Seconds
--|-------|--------
 1 | 233479887 | 23
 2 | 132001706 | 13
 3 |  98215813 |  9
 4 | 105532869 | 10
 5 |  95057497 |  9
 6 |  87390268 |  8
 7 |  89011266 |  8
 8 |  91075460 |  9
 9 |  90258810 |  9
10 |  89722936 |  8
11 |  91529955 |  9
12 |  91633545 |  9

# Еквівалентність

Отже, в межах цього проєкту ми розглянули два підходи до вирішення однієї задачі.
Важливо перевірити, що дані підходи справді мають аналогічні результати.

```PowerShell
PS> mpiexec.exe -n 4 .\bin\Release\MPI.exe "C:\Users\Fyodor\The Legend of Vox Machina" > mpi.txt
PS> .\bin\Release\OMP.exe 4 "C:\Users\Fyodor\The Legend of Vox Machina" > omp.txt
```

Порівнюючи вміст файлів `mpi.txt` та `omp.txt`, ми й справді можемо переконатися, що одні й ті самі файли отримують одні й ті самі хеші.

# MPI vs OpenMP

Також цікаво порівняти швидкодію MPI та OpenMP.
Пропонуємо розглянути наступну звідну таблицю.

n | MPI<br>Debug | OpenMP<br>Debug | **Diff** | MPI<br>Release | OpenMP<br>Release | **Diff**
-:|-:|-:|-:|-:|-:|-:
 1 | 28 | 23 | -5 | 24 | 23 | -1
 2 | 13 | 14 | +1 | 13 | 13 |
 3 | 10 | 11 | +1 |  9 |  9 |
 4 |  8 | 11 | +3 |  7 | 10 | +3
 5 |  7 | 11 | +4 |  8 |  9 | +1
 6 | 10 | 10 |    |  9 |  8 | -1
 7 |  8 | 11 | +3 |  7 |  8 | +1
 8 |  9 | 10 | +1 |  8 |  9 | +1
 9 |  9 | 10 | +1 |  8 |  9 | +1
10 |  9 | 10 | +1 |  8 |  8 |
11 |  9 | 10 | +1 |  8 |  9 | +1
12 |  9 | 10 | +1 |  8 |  9 | +1