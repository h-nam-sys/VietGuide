# Explicit operations setup, not a source installer. Run only on the deployment PC.
param(
    [ValidateSet('Logon','Boot','Remove')][string]$Mode = 'Logon',
    [string]$Project = (Split-Path $PSScriptRoot -Parent),
    [string]$TaskName = 'LegalLLM-Services'
)
$ErrorActionPreference = 'Stop'
if ($Mode -eq 'Remove') {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    exit
}
$Project = (Resolve-Path $Project).Path
$Python = Join-Path $Project 'env\Scripts\python.exe'
if (!(Test-Path $Python)) { throw 'Project env\Scripts\python.exe was not found' }
$Action = New-ScheduledTaskAction -Execute $Python -Argument ('-u "' + (Join-Path $Project 'run_services.py') + '" --production') -WorkingDirectory $Project
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew
if ($Mode -eq 'Logon') {
    $User = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
    $Trigger = New-ScheduledTaskTrigger -AtLogOn -User $User
    $Principal = New-ScheduledTaskPrincipal -UserId $User -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Principal $Principal -Settings $Settings
} else {
    # A real boot task needs a Windows account credential. Do not use SYSTEM:
    # GPU drivers, model caches and environment must be verified for this account.
    $Credential = Get-Credential -Message 'Deployment account with access to model files and the GPU'
    $Trigger = New-ScheduledTaskTrigger -AtStartup
    $Password = $Credential.GetNetworkCredential().Password
    try {
        Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -User $Credential.UserName -Password $Password
    } finally {
        $Password = $null
        $Credential = $null
    }
}
Write-Output 'Task registered. Reboot/logon and GPU readiness must be verified on this machine.'
