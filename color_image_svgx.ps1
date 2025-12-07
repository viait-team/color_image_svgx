$Global:IMAGEMAGICK_CONVERT_PATH = "magick.exe"
$Global:POTRACE_PATH = "potrace.exe"


# DPI setting for Potrace output scaling
$Global:POTRACE_DPI = 90.0  # Potrace docs say 72, but 90 works best in your pipeline

# Threshold for command-line length before splitting
$Global:COMMAND_LEN_NEAR_MAX = 1900  # Conservative limit for large command strings

# Verbosity level for logging and diagnostics
$Global:VERBOSITY_LEVEL = 0  # Can be overridden by -v or --verbose flags


function Invoke-ProcessCommand {
    param (
        [string]$Command,            # Full command string with arguments
        [object]$StdInput = $null,   # Optional input to send to stdin
        [bool]$CaptureStdOut = $false,
        [bool]$CaptureStdErr = $false
    )

    Write-Verbose "Running: $Command"

    $stdoutPath = if ($CaptureStdOut) { [System.IO.Path]::GetTempFileName() } else { $null }
    $stderrPath = if ($CaptureStdErr) { [System.IO.Path]::GetTempFileName() } else { $null }
    $stdinPath = if ($StdInput) { [System.IO.Path]::GetTempFileName() } else { $null }

    if ($StdInput) {
        if ($StdInput -is [byte[]]) {
            [System.IO.File]::WriteAllBytes($stdinPath, $StdInput)
        }
        else {
            Set-Content -Path $stdinPath -Value $StdInput -Encoding UTF8
        }
    }
    $redirects = @()
    if ($stdoutPath) { $redirects += "1> `"$stdoutPath`"" }
    if ($stderrPath) { $redirects += "2> `"$stderrPath`"" }

    $stdinRedirect = if ($stdinPath) { "< `"$stdinPath`"" } else { "" }
    $fullCommand = "$Command $stdinRedirect $($redirects -join ' ')"

    $exitCode = Invoke-Expression $fullCommand
    $LASTEXITCODE = $global:LASTEXITCODE

    if ($LASTEXITCODE -ne 0) {
        $errMsg = if ($stderrPath) { Get-Content $stderrPath -Raw } else { "Process failed with exit code $LASTEXITCODE" }
        throw $errMsg
    }

    $stdout = if ($stdoutPath) { Get-Content $stdoutPath -Raw } else { $null }
    $stderr = if ($stderrPath) { Get-Content $stderrPath -Raw } else { $null }

    # Cleanup temp files
    foreach ($path in @($stdoutPath, $stderrPath, $stdinPath)) {
        if ($path -and (Test-Path $path)) { Remove-Item $path -Force }
    }

    if ($CaptureStdOut -and -not $CaptureStdErr) {
        return $stdout
    }
    elseif ($CaptureStdErr -and -not $CaptureStdOut) {
        return $stderr
    }
    elseif ($CaptureStdOut -and $CaptureStdErr) {
        return @{ StdOut = $stdout; StdErr = $stderr }
    }
    else {
        return $null
    }
}

function Rescale-Image {
    param (
        [string]$Src,
        [string]$DestScale,
        [double]$Scale,
        [string]$Filter = 'lanczos'
    )

    if ($Scale -eq 1.0) {
        Copy-Item -Path $Src -Destination $DestScale -Force
        return
    }

    $ConvertPath = $Global:IMAGEMAGICK_CONVERT_PATH
    $ResizePercent = [math]::Round($Scale * 100, 2)
    $Command = '"{0}" "{1}" -filter {2} -resize {3}% "{4}"' -f $ConvertPath, $Src, $Filter, $ResizePercent, $DestScale

    Invoke-ProcessCommand -Command $Command
}

function Quantize-Image {
    param (
        [string]$Src,
        [string]$DestQuant,
        [int]$Colors,
        [string]$Algorithm = 'mc',
        [string]$Dither = $null
    )

    if ($Colors -eq 0) {
        Copy-Item -Path $Src -Destination $DestQuant -Force
        return
    }

    switch ($Algorithm) {
        'mc' {
            switch ($Dither) {
                $null { $DitherOpt = '-nofs ' }
                'floydsteinberg' { $DitherOpt = '' }
                default { throw "Invalid dither type '$Dither' for 'mc' quantization." }
            }

            $PngQuantPath = $Global:PNGQUANT_PATH
            $Command = '"{0}" {1}-force {2}' -f $PngQuantPath, $DitherOpt, $Colors

            $StdInput = Get-Content -Path $Src -Encoding Byte
            $StdOutput = Invoke-ProcessCommand -Command $Command -StdInput $StdInput -CaptureStdOut $true
            [System.IO.File]::WriteAllBytes($DestQuant, $StdOutput)
        }

        'as' {
            switch ($Dither) {
                $null { $DitherOpt = 'None' }
                'floydsteinberg' { $DitherOpt = 'floydsteinberg' }
                'riemersma' { $DitherOpt = 'riemersma' }
                default { throw "Invalid dither type '$Dither' for 'as' quantization." }
            }

            $ConvertPath = $Global:IMAGEMAGICK_CONVERT_PATH
            $Command = '"{0}" "{1}" -dither {2} -colors {3} "{4}"' -f $ConvertPath, $Src, $DitherOpt, $Colors, $DestQuant
            Invoke-ProcessCommand -Command $Command
        }

        'nq' {
            $Ext = "~quant.png"
            $DestDir = Split-Path $DestQuant -Parent
            switch ($Dither) {
                $null { $DitherOpt = '' }
                'floydsteinberg' { $DitherOpt = '-Q f ' }
                default { throw "Invalid dither type '$Dither' for 'nq' quantization." }
            }

            $PngNqPath = $Global:PNGNQ_PATH
            $Command = '"{0}" -f {1}-d "{2}" -n {3} -e {4} "{5}"' -f $PngNqPath, $DitherOpt, $DestDir, $Colors, $Ext, $Src
            Invoke-ProcessCommand -Command $Command

            $BaseName = [System.IO.Path]::GetFileNameWithoutExtension($Src)
            $OldDest = Join-Path $DestDir "$BaseName$Ext"
            Move-Item -Path $OldDest -Destination $DestQuant -Force
        }

        default {
            throw "Unknown quantization algorithm '$Algorithm'"
        }
    }
}

function Remap-ImagePalette {
    param (
        [string]$Src,
        [string]$DestRemap,
        [string]$PaletteImage,
        [string]$Dither = $null
    )

    if (-not (Test-Path $PaletteImage)) {
        throw "Remapping palette image '$PaletteImage' not found."
    }

    switch ($Dither) {
        $null { $DitherOpt = 'None' }
        'floydsteinberg' { $DitherOpt = 'floydsteinberg' }
        'riemersma' { $DitherOpt = 'riemersma' }
        default { throw "Invalid dither type '$Dither' for remapping." }
    }

    $ConvertPath = $Global:IMAGEMAGICK_CONVERT_PATH
    $Command = '"{0}" "{1}" -dither {2} -remap "{3}" "{4}"' -f $ConvertPath, $Src, $DitherOpt, $PaletteImage, $DestRemap

    Invoke-ProcessCommand -Command $Command
}

function Get-PaletteFromImage {
    param (
        [string]$SrcImage
    )

    $ConvertPath = $Global:IMAGEMAGICK_CONVERT_PATH
    $Command = '"{0}" "{1}" -unique-colors -compress none ppm:-' -f $ConvertPath, $SrcImage

    $StdOutput = Invoke-ProcessCommand -Command $Command -CaptureStdOut $true
    $ppmLines = $StdOutput -split "`n"
    $ppmLines = $ppmLines[3..($ppmLines.Count - 1)]  # Skip PPM header

    Remove-Variable StdOutput -ErrorAction SilentlyContinue  # Free memory

    $ColorVals = @()
    foreach ($line in $ppmLines) {
        $ColorVals += ($line -split '\s+') | ForEach-Object { [int]$_ }
    }

    $HexColors = @()
    for ($i = 0; $i -lt $ColorVals.Count; $i += 3) {
        $r = $ColorVals[$i]
        $g = $ColorVals[$i + 1]
        $b = $ColorVals[$i + 2]
        $HexColors += ('#{0:X2}{1:X2}{2:X2}' -f $r, $g, $b).ToLower()
    }

    [array]::Reverse($HexColors)
    return $HexColors
}


function Get-NonPaletteColor {
    param (
        [string[]]$Palette,
        [bool]$StartBlack = $true,
        [string[]]$Additional = @()
    )

    # Combine palette and additional exclusions
    $PaletteSet = $Palette + $Additional
    $PaletteHash = @{}
    foreach ($color in $PaletteSet) {
        $PaletteHash[$color.ToLower()] = $true
    }

    # Define search range
    if ($StartBlack) {
        $ColorRange = 0..0xFFFFFF
    } else {
        $ColorRange = 0xFFFFFF..0
    }

    foreach ($i in $ColorRange) {
        $Hex = "#{0:X6}" -f $i
        if (-not $PaletteHash.ContainsKey($Hex.ToLower())) {
            return $Hex
        }
    }

    throw "All colors exhausted — could not find a nonpalette color."
}

function Isolate-Color {
    param (
        [string]$Src,
        [string]$TargetTmp,
        [string]$DestLayer,
        [string]$TargetColor,
        [string[]]$Palette,
        [bool]$Stack = $false
    )

    $ColorIdx = ($Palette | ForEach-Object -Begin { $i = 0 } -Process {
            if ($_ -ieq $TargetColor) { $i }
            $i++
        }) | Select-Object -First 1

    $BgWhite = "#FFFFFF"
    $FgBlack = "#000000"
    $BgAlmostWhite = Get-NonPaletteColor -Palette $Palette -StartBlack $false -Additional @($BgWhite, $FgBlack)
    $FgAlmostBlack = Get-NonPaletteColor -Palette $Palette -StartBlack $true -Additional @($BgAlmostWhite, $BgWhite, $FgBlack)

    $StdInput = Get-Content -Path $Src -Encoding Byte

    $CommandPre = "`"$IMAGEMAGICK_CONVERT_PATH`" `"$Src`""
    $CommandPost = "`"$TargetTmp`""
    $CommandMid = ""
    $LastIteration = $Palette.Count - 1

    foreach ($i in 0..$LastIteration) {
        $Col = $Palette[$i]
        if ($i -eq $ColorIdx -or ($Stack -and $i -gt $ColorIdx)) {
            $Fill = $FgAlmostBlack
        } else {
            $Fill = $BgAlmostWhite
        }

        $CommandMid += " -fill `"$Fill`" -opaque `"$Col`""

        if ($CommandMid.Length -ge $Global:COMMAND_LEN_NEAR_MAX -or ($i -eq $LastIteration -and $CommandMid)) {
            $FullCommand = "$CommandPre $CommandMid $CommandPost"
            $StdOutput = Invoke-ProcessCommand -Command $FullCommand -StdInput $StdInput
            $StdInput = $StdOutput
            $CommandMid = ""
        }
    }

    $FinalCommand = "`"$IMAGEMAGICK_CONVERT_PATH`" `"$TargetTmp`" -fill `"$BgWhite`" -opaque `"$BgAlmostWhite`" -fill `"$FgBlack`" -opaque `"$FgAlmostBlack`" `"$DestLayer`""
    Invoke-ProcessCommand  -Command $FinalCommand -StdInput $StdInput
}

function Fill-WithColor {
    param (
        [string]$Src,
        [string]$Dest
    )

    $Color = "#000000"
    $Command = "`"$IMAGEMAGICK_CONVERT_PATH`" `"$Src`" -fill `"$Color`" -opaque none `"$Dest`""
    Invoke-ProcessCommand -Command $Command
}

function Get-ImageWidth {
    param (
        [string]$Src
    )

    $Command = "`"$IMAGEMAGICK_CONVERT_PATH`" -ping -format `%w `"$Src`""
    $StdOutput = Invoke-ProcessCommand  -Command $Command -CaptureStdOut $true
    $Width = [int]$StdOutput.Trim()
    return $Width
}

function trace {
    param (
        [string]$Src,
        [string]$DestTrace,
        [string]$OutColor,
        [int]$Despeckle = 2,
        [double]$SmoothCorners = 1.0,
        [double]$OptimizePaths = 0.2,
        [double]$Width = $null
    )

    if ($Width -ne $null) {
        $ScaledWidth = $Width / $Global:POTRACE_DPI
        $WidthArg = "-W $ScaledWidth"
    }
    else {
        $WidthArg = ""
    }

    $Command = "`"$POTRACE_PATH`" --svg -o `"$DestTrace`" -C `"$OutColor`" -t $Despeckle -a $SmoothCorners -O $OptimizePaths $WidthArg `"$Src`""
    Invoke-Expression $Command
}

function check_range {
    param (
        [double]$Min,
        [double]$Max = [double]::PositiveInfinity,
        [ScriptBlock]$TypeFunc,
        [string]$TypeName,
        [string]$StrVal
    )

    try {
        $Val = & $TypeFunc $StrVal
    }
    catch {
        throw "Value '$StrVal' must be $TypeName."
    }

    if ($Val -lt $Min) {
        throw "Value '$Val' must be $Min or greater."
    }

    if ($Max -ne [double]::PositiveInfinity -and $Val -gt $Max) {
        throw "Value '$Val' must be between $Min and $Max."
    }

    return $Val
}