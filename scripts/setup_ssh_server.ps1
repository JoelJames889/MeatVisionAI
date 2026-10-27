# ==============================================================================
# Script: setup_ssh_server.ps1
# Description: Installs and configures OpenSSH Server on Windows with firewall
#              and correct permissions for administrator accounts.
# ==============================================================================

# Ensure script is running with elevated privileges
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "[!] Administrator privileges are required to install OpenSSH Server and configure the Windows Firewall." -ForegroundColor Yellow
    Write-Host "    Attempting to launch elevated PowerShell prompt..." -ForegroundColor Cyan
    try {
        Start-Process powershell.exe -ArgumentList ("-NoProfile -ExecutionPolicy Bypass -File `"{0}`"" -f $MyInvocation.MyCommand.Path) -Verb RunAs -ErrorAction Stop
        Exit
    } catch {
        Write-Host "`n[!] Could not automatically pop up UAC prompt (running in non-interactive session)." -ForegroundColor Red
        Write-Host "`nPlease open an Administrator PowerShell window and run:" -ForegroundColor Yellow
        Write-Host "powershell -ExecutionPolicy Bypass -File `"$($MyInvocation.MyCommand.Path)`"" -ForegroundColor Green
        Exit 1
    }
}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "       Setting up Windows OpenSSH Server for MeatVision   " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Install OpenSSH Server capability
Write-Host "`n[1/5] Checking OpenSSH Server capability..." -ForegroundColor Green
$sshCapability = Get-WindowsCapability -Online | Where-Object Name -like 'OpenSSH.Server*'

if ($sshCapability.State -ne 'Installed') {
    Write-Host "      Installing OpenSSH.Server capability..." -ForegroundColor Yellow
    Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
    Write-Host "      OpenSSH.Server installed successfully." -ForegroundColor Green
} else {
    Write-Host "      OpenSSH Server is already installed." -ForegroundColor Green
}

# 2. Configure and start Services
Write-Host "`n[2/5] Configuring sshd and ssh-agent services..." -ForegroundColor Green
Set-Service -Name sshd -StartupType 'Automatic'
Start-Service -Name sshd -ErrorAction SilentlyContinue

Set-Service -Name ssh-agent -StartupType 'Automatic'
Start-Service -Name ssh-agent -ErrorAction SilentlyContinue
Write-Host "      sshd and ssh-agent services configured and running." -ForegroundColor Green

# 3. Configure Windows Firewall
Write-Host "`n[3/5] Configuring Windows Defender Firewall..." -ForegroundColor Green
$fwRule = Get-NetFirewallRule -Name "OpenSSH-Server-In-TCP" -ErrorAction SilentlyContinue
if (-not $fwRule) {
    New-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -DisplayName 'OpenSSH Server (sshd)' -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22
    Write-Host "      Firewall rule created for port 22 (TCP)." -ForegroundColor Green
} else {
    Enable-NetFirewallRule -Name "OpenSSH-Server-In-TCP"
    Write-Host "      Firewall rule for port 22 verified and enabled." -ForegroundColor Green
}

# 4. Configure Default Shell to PowerShell
Write-Host "`n[4/5] Setting default OpenSSH shell to PowerShell..." -ForegroundColor Green
if (-not (Test-Path "HKLM:\SOFTWARE\OpenSSH")) {
    New-Item -Path "HKLM:\SOFTWARE\OpenSSH" -Force | Out-Null
}
New-ItemProperty -Path "HKLM:\SOFTWARE\OpenSSH" -Name DefaultShell -Value "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -PropertyType String -Force | Out-Null

# 5. Fix administrator authorized_keys configuration
Write-Host "`n[5/5] Configuring sshd_config & user .ssh directory..." -ForegroundColor Green
$sshdConfigPath = "$env:ProgramData\ssh\sshd_config"
if (Test-Path $sshdConfigPath) {
    $content = Get-Content $sshdConfigPath -Raw
    # Comment out the force override for administrators_authorized_keys so ~/.ssh/authorized_keys works for admin accounts
    if ($content -match "Match Group administrators") {
        $updated = $content -replace "(?m)^(\s*Match Group administrators)", "#`$1" `
                            -replace "(?m)^(\s*AuthorizedKeysFile __PROGRAMDATA__/ssh/administrators_authorized_keys)", "#`$1"
        if ($updated -ne $content) {
            Set-Content -Path $sshdConfigPath -Value $updated -Encoding UTF8
            Restart-Service sshd -Force
            Write-Host "      Updated sshd_config to allow standard ~/.ssh/authorized_keys." -ForegroundColor Green
        }
    }
}

# Ensure user .ssh folder exists
$userSshDir = "$env:USERPROFILE\.ssh"
if (-not (Test-Path $userSshDir)) {
    New-Item -ItemType Directory -Path $userSshDir -Force | Out-Null
}
$authKeysPath = "$userSshDir\authorized_keys"
if (-not (Test-Path $authKeysPath)) {
    New-Item -ItemType File -Path $authKeysPath -Force | Out-Null
}

# Get current IPv4 addresses
$ips = Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.InterfaceAlias -notlike "*Loopback*" -and $_.IPAddress -notlike "169.254*" } | Select-Object -ExpandProperty IPAddress

Write-Host "`n==========================================================" -ForegroundColor Cyan
Write-Host "                 SSH SETUP COMPLETE!                      " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Username   : $env:USERNAME" -ForegroundColor Yellow
Write-Host "Password   : (Your Windows login password)" -ForegroundColor Yellow
Write-Host "Local IPs  : $($ips -join ', ')" -ForegroundColor Yellow
Write-Host "`nHow to connect from another computer on the same network:" -ForegroundColor White
foreach ($ip in $ips) {
    Write-Host "   ssh $env:USERNAME@$ip" -ForegroundColor Green
}
Write-Host "`nTo use key-based authentication, paste your public key into:" -ForegroundColor White
Write-Host "   $authKeysPath" -ForegroundColor DarkGray
Write-Host "==========================================================`n" -ForegroundColor Cyan

Pause
