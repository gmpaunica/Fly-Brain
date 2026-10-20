<#
.SYNOPSIS
Starts an interactive Sol session for this project without changing saved settings.
.EXAMPLE
.\scripts\Start-Sol.ps1 -Effort high
.EXAMPLE
.\scripts\Start-Sol.ps1 -Preview
#>
[CmdletBinding()]
param(
    [ValidateSet('medium', 'high')]
    [string]$Effort = 'medium',
    [switch]$Preview
)

$ErrorActionPreference = 'Stop'
$flyProjectRoot = Split-Path -Parent $PSScriptRoot
$flyPrompt = 'Read AGENTS.md and PROJECT_STATUS.md, then the latest relevant project log entry. Implement only the active task through its acceptance checks and update the project records. Work with one agent. Make routine implementation decisions yourself. If a task requires a material user decision, state it precisely.'
$flyArguments = @(
    '--cd', $flyProjectRoot,
    '--model', 'gpt-5.6-sol',
    '-c', ('model_reasoning_effort="{0}"' -f $Effort),
    $flyPrompt
)

if ($Preview) {
    [pscustomobject]@{ Executable = 'codex'; Arguments = $flyArguments }
    return
}

$null = Get-Command codex -ErrorAction Stop
& codex @flyArguments
if ($LASTEXITCODE -ne 0) {
    throw "Codex exited with code $LASTEXITCODE."
}
