#!/usr/bin/env sh

# scripts/initialize-admin-database.sh :
# This script initializes the admin database by creating the necessary tables and inserting the admin user.

duckdb \
    -init ~/.local/share/pfun-cma-model/admin_create_user.sql \
    ~/.local/share/pfun-cma-model/admin.db 