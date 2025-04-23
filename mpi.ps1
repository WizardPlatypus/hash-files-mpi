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
