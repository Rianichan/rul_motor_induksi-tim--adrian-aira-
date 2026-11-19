param(
    [Parameter(Mandatory=$true)][string]$CsvPath,
    [string]$OutputDir = (Get-Location).Path
)

$ErrorActionPreference = 'Stop'
$featureColumns = @('beban_pct','arus_A','tegangan_V','suhu_belitan_C','getaran_rms_mm_s','kecepatan_rpm')

function Get-CanonicalMotorId([string]$Value) {
    return (($Value.Trim() -replace '[^A-Za-z0-9]', '').ToUpperInvariant())
}

function Get-CanonicalType([string]$Value) {
    $v = (($Value.Trim() -replace '\s', '') -replace ',', '.').ToLowerInvariant()
    switch ($v) {
        '5.5kw' { return '5.5kW' }
        '11kw'  { return '11kW' }
        '22kw'  { return '22kW' }
        default { return 'UNKNOWN' }
    }
}

function Get-Median([double[]]$Values) {
    if ($Values.Count -eq 0) { return 0.0 }
    $sorted = @($Values | Sort-Object)
    $middle = [int][Math]::Floor($sorted.Count / 2)
    if (($sorted.Count % 2) -eq 1) { return [double]$sorted[$middle] }
    return (([double]$sorted[$middle - 1] + [double]$sorted[$middle]) / 2.0)
}

function Get-FeatureVector($Row, $Impute, $Means, $Stds) {
    $x = [double[]]::new(9)
    $x[0] = 1.0
    for ($k=0; $k -lt $featureColumns.Count; $k++) {
        $col = $featureColumns[$k]
        $raw = $Row.$col
        if ($col -eq 'getaran_rms_mm_s' -and $raw -ne '' -and [double]::Parse($raw,[Globalization.CultureInfo]::InvariantCulture) -lt 0) { $raw = '' }
        if ([string]::IsNullOrWhiteSpace($raw)) { $value = [double]$Impute[$col] }
        else { $value = [double]::Parse($raw,[Globalization.CultureInfo]::InvariantCulture) }
        $x[$k + 1] = ($value - [double]$Means[$col]) / [double]$Stds[$col]
    }
    if ($Row.tipe_motor -eq '11kW') { $x[7] = 1.0 }
    if ($Row.tipe_motor -eq '22kW') { $x[8] = 1.0 }
    return ,$x
}

if (-not (Test-Path -LiteralPath $OutputDir)) { New-Item -ItemType Directory -Path $OutputDir | Out-Null }
$rawRows = Import-Csv -LiteralPath $CsvPath
$seen = [System.Collections.Generic.HashSet[string]]::new()
$rows = [System.Collections.Generic.List[object]]::new()
foreach ($r in $rawRows) {
    $r.motor_id = Get-CanonicalMotorId $r.motor_id
    $r.tipe_motor = Get-CanonicalType $r.tipe_motor
    $rowKey = ($r.PSObject.Properties.Value -join [char]31)
    if (-not $seen.Add($rowKey)) { continue }
    $rows.Add($r)
}

# Reproduce the EDA split: sorted canonical IDs, seeded Fisher–Yates, 70/15/15.
$motorIds = @($rows | ForEach-Object motor_id | Sort-Object -Unique)
$rng = [Random]::new(42)
for ($i = $motorIds.Count - 1; $i -gt 0; $i--) {
    $j = $rng.Next($i + 1)
    $tmp = $motorIds[$i]; $motorIds[$i] = $motorIds[$j]; $motorIds[$j] = $tmp
}
$trainIds = [System.Collections.Generic.HashSet[string]]::new([string[]]$motorIds[0..209])
$validationIds = [System.Collections.Generic.HashSet[string]]::new([string[]]$motorIds[210..254])
$testIds = [System.Collections.Generic.HashSet[string]]::new([string[]]$motorIds[255..299])
$manifest = foreach ($id in $trainIds) { [pscustomobject]@{motor_id=$id; split='train'} }
$manifest += foreach ($id in $validationIds) { [pscustomobject]@{motor_id=$id; split='validation'} }
$manifest += foreach ($id in $testIds) { [pscustomobject]@{motor_id=$id; split='test'} }
$manifest | Sort-Object motor_id | Export-Csv -LiteralPath (Join-Path $OutputDir 'split_motor_kasus03.csv') -NoTypeInformation -Encoding UTF8

$trainRows = @($rows | Where-Object { $trainIds.Contains($_.motor_id) })
$validationRows = @($rows | Where-Object { $validationIds.Contains($_.motor_id) })

# Infer each motor's fixed nominal type from its own split only.
foreach ($partition in @($trainRows, $validationRows)) {
    foreach ($g in ($partition | Group-Object motor_id)) {
        $mode = $g.Group | Group-Object tipe_motor | Sort-Object Count -Descending | Select-Object -First 1 -ExpandProperty Name
        foreach ($r in $g.Group) { $r.tipe_motor = $mode }
    }
}

# Fit imputation and scaling parameters on train rows only.
$impute = @{}
$means = @{}
$stds = @{}
foreach ($col in $featureColumns) {
    $vals = @($trainRows | ForEach-Object {
        $raw = $_.$col
        if ($col -eq 'getaran_rms_mm_s' -and $raw -ne '' -and [double]::Parse($raw,[Globalization.CultureInfo]::InvariantCulture) -lt 0) { return }
        if (-not [string]::IsNullOrWhiteSpace($raw)) { [double]::Parse($raw,[Globalization.CultureInfo]::InvariantCulture) }
    })
    $impute[$col] = Get-Median $vals
    $filled = @($trainRows | ForEach-Object {
        $raw = $_.$col
        if ($col -eq 'getaran_rms_mm_s' -and $raw -ne '' -and [double]::Parse($raw,[Globalization.CultureInfo]::InvariantCulture) -lt 0) { $raw = '' }
        if ([string]::IsNullOrWhiteSpace($raw)) { [double]$impute[$col] }
        else { [double]::Parse($raw,[Globalization.CultureInfo]::InvariantCulture) }
    })
    $sum = 0.0; foreach ($v in $filled) { $sum += [double]$v }
    $mean = $sum / $filled.Count
    $sq = 0.0; foreach ($v in $filled) { $delta = [double]$v - $mean; $sq += $delta * $delta }
    $std = [Math]::Sqrt($sq / $filled.Count)
    if ($std -eq 0) { $std = 1.0 }
    $means[$col] = $mean; $stds[$col] = $std
}

# Ordinary least squares via normal equations with partial-pivot Gauss–Jordan.
$p = 9
$xtx = [System.Collections.Generic.List[double[]]]::new()
for ($i=0; $i -lt $p; $i++) { $xtx.Add([double[]]::new($p)) }
$xty = [double[]]::new($p)
foreach ($r in $trainRows) {
    $x = Get-FeatureVector $r $impute $means $stds
    $y = [double]::Parse($r.rul_hari,[Globalization.CultureInfo]::InvariantCulture)
    for ($i=0; $i -lt $p; $i++) {
        $xty[$i] += $x[$i] * $y
        for ($j=0; $j -lt $p; $j++) { $xtx[$i][$j] += $x[$i] * $x[$j] }
    }
}
for ($k=0; $k -lt $p; $k++) {
    $pivot = $k
    for ($i=$k + 1; $i -lt $p; $i++) { if ([Math]::Abs($xtx[$i][$k]) -gt [Math]::Abs($xtx[$pivot][$k])) { $pivot = $i } }
    if ([Math]::Abs($xtx[$pivot][$k]) -lt 1e-12) { throw "Singular design matrix near column $k" }
    if ($pivot -ne $k) { $tmp=$xtx[$k]; $xtx[$k]=$xtx[$pivot]; $xtx[$pivot]=$tmp; $t=$xty[$k]; $xty[$k]=$xty[$pivot]; $xty[$pivot]=$t }
    $divisor = $xtx[$k][$k]
    for ($j=$k; $j -lt $p; $j++) { $xtx[$k][$j] /= $divisor }
    $xty[$k] /= $divisor
    for ($i=0; $i -lt $p; $i++) {
        if ($i -eq $k) { continue }
        $factor = $xtx[$i][$k]
        for ($j=$k; $j -lt $p; $j++) { $xtx[$i][$j] -= $factor * $xtx[$k][$j] }
        $xty[$i] -= $factor * $xty[$k]
    }
}
$coefficients = $xty

$absoluteErrors = [System.Collections.Generic.List[double]]::new()
$criticalErrors = [System.Collections.Generic.List[double]]::new()
$clippedPredictions = 0
foreach ($r in $validationRows) {
    $x = Get-FeatureVector $r $impute $means $stds
    $pred = 0.0
    for ($i=0; $i -lt $p; $i++) { $pred += $x[$i] * $coefficients[$i] }
    if ($pred -lt 0) { $pred = 0.0; $clippedPredictions++ }
    $actual = [double]::Parse($r.rul_hari,[Globalization.CultureInfo]::InvariantCulture)
    $err = [Math]::Abs($actual - $pred)
    $absoluteErrors.Add($err)
    if ($actual -le 60) { $criticalErrors.Add($err) }
}
$maeAll = ($absoluteErrors | Measure-Object -Average).Average
$maeCritical = ($criticalErrors | Measure-Object -Average).Average

$featureNames = @('intercept','beban_pct','arus_A','tegangan_V','suhu_belitan_C','getaran_rms_mm_s','kecepatan_rpm','tipe_11kW','tipe_22kW')
$result = [ordered]@{
    model = 'ordinary_least_squares_linear_regression'
    split_seed = 42
    split_motors = [ordered]@{train=210; validation=45; test_held_out=45}
    feature_names = $featureNames
    coefficients = @($coefficients)
    numeric_means_train = $means
    numeric_stds_train = $stds
    median_imputation_train = $impute
    categorical_reference = '5.5kW'
    metrics_validation = [ordered]@{
        rows_all = $absoluteErrors.Count
        rows_rul_le_60 = $criticalErrors.Count
        mae_all_days = [Math]::Round([double]$maeAll, 4)
        mae_rul_le_60_days = [Math]::Round([double]$maeCritical, 4)
        negative_raw_predictions_clipped_to_zero = $clippedPredictions
    }
}
$modelPath = Join-Path $OutputDir 'baseline_linear_kasus03.json'
$result | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $modelPath -Encoding UTF8
$result.metrics_validation | ConvertTo-Json -Depth 4
Write-Output ('MODEL=' + $modelPath)
Write-Output ('SPLIT_MANIFEST=' + (Join-Path $OutputDir 'split_motor_kasus03.csv'))
