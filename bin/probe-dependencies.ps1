param([string]$OutputPath = '')
$ErrorActionPreference = 'Stop'
$packageNames = @('textual','aiohttp','psutil','platformdirs','opentelemetry-api','opentelemetry-sdk','opentelemetry-exporter-otlp-proto-grpc','pytest','pytest-asyncio','build','hatchling','ruff','yt-dlp','yt-dlp-ejs')
$packageFacts = foreach ($packageName in $packageNames) {
    $source = 'https://pypi.org/pypi/' + $packageName + '/json'
    $packageInfo = Invoke-RestMethod -Uri $source -TimeoutSec 30
    [pscustomobject]@{name=$packageName;version=$packageInfo.info.version;requires_python=$packageInfo.info.requires_python;source=$source}
}
$json = $packageFacts | ConvertTo-Json
if ($OutputPath) { $json | Set-Content -LiteralPath $OutputPath -Encoding utf8 }
$packageFacts | Format-Table name,version,requires_python
