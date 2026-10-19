#!/bin/sh
# MariaDB login for slurmdbd: unix_socket authentication for the OS user `slurm` (as on the box), full rights on its database.
set -eu
mariadb <<'SQL'
CREATE USER IF NOT EXISTS 'slurm'@'localhost' IDENTIFIED VIA unix_socket;
GRANT ALL ON slurm_acct_db.* TO 'slurm'@'localhost';
SQL
