#!/usr/bin/env python3
"""
generate-dev-certs.py
Script to generate self-signed development certificates using Python.
Uses Click for CLI and Textual for interactive menu.
"""

import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import click
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Footer, Header, RadioButton, RadioSet


class CertGeneratorApp(App):
    """Textual app for selecting certificate generation method."""

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            RadioSet(
                RadioButton(
                    "Tailscale (Letsencrypt, served internal on the tailnet)",
                    id="tailscale",
                ),
                RadioButton("OpenSSL (self-signed, local)", id="openssl"),
                id="cert_method",
            ),
            Button("Generate", id="generate", variant="primary"),
            id="main_container",
        )
        yield Footer()

    def on_button_pressed(self, event):
        if event.button.id == "generate":
            radio_set = self.query_one("#cert_method", RadioSet)
            selected = radio_set.pressed_button
            if selected:
                method = selected.id
                self.exit(method)
            else:
                self.exit(None)


@click.command()
@click.option(
    "--tailscale",
    is_flag=True,
    help="Generate Tailscale certificates without interactive prompt.",
)
@click.option(
    "--self-signed",
    is_flag=True,
    help="Generate self-signed certificates without interactive prompt.",
)
def main(tailscale, self_signed):
    """Generate development certificates."""
    # Validate that only one flag is provided
    if tailscale and self_signed:
        click.echo(
            "Error: Cannot specify both --tailscale and --self-signed flags. "
            "Please choose one method.",
            err=True,
        )
        sys.exit(1)

    click.echo("Generating self-signed development certificates...")

    certs_dir = Path("./certs")
    certs_dir.mkdir(exist_ok=True)

    # If neither flag is provided, show interactive menu
    if not tailscale and not self_signed:
        app = CertGeneratorApp()
        choice = app.run()
    else:
        # Use flag to determine choice
        if tailscale:
            choice = "tailscale"
        else:
            choice = "openssl"

    if choice == "tailscale":
        generate_certs_tailscale(certs_dir)
    elif choice == "openssl":
        generate_certs_openssl(certs_dir)
    else:
        click.echo("No method selected.", err=True)
        sys.exit(1)

    click.echo(f"Certificates generated in {certs_dir}/")


def generate_certs_openssl(certs_dir: Path):
    """Generate self-signed certificate with OpenSSL."""
    click.echo("Generating self-signed certificate with OpenSSL...")

    key_path = certs_dir / "key.pem"
    cert_path = certs_dir / "cert.pem"

    cmd = [
        "openssl",
        "req",
        "-x509",
        "-newkey",
        "rsa:4096",
        "-keyout",
        str(key_path),
        "-out",
        str(cert_path),
        "-days",
        "365",
        "-nodes",
        "-subj",
        "/CN=localhost/O=Development",
    ]

    try:
        subprocess.run(cmd, check=True)
        click.echo("Done.")
    except subprocess.CalledProcessError as e:
        click.echo(f"Error generating OpenSSL certificates: {e}", err=True)
        sys.exit(1)


def generate_certs_tailscale(certs_dir: Path):
    """Generate certificate with Tailscale."""
    if not shutil.which("tailscale"):
        click.echo(
            "Error: tailscale command not found. Install Tailscale first.", err=True
        )
        sys.exit(1)

    hostname = socket.gethostname()
    domain_arg = f"{hostname}.tail38611b.ts.net"

    click.echo("Generating certificate with Tailscale...")
    click.echo(f"  Host: '{domain_arg}'")
    click.echo("  Note: Ensure you are logged into Tailscale.")

    cert_file = certs_dir / f"{domain_arg}.crt"
    key_file = certs_dir / f"{domain_arg}.key"

    cmd = [
        "tailscale",
        "cert",
        "--cert-file",
        str(cert_file),
        "--key-file",
        str(key_file),
        domain_arg,
    ]

    try:
        subprocess.run(cmd, check=True)
        click.echo("Done.")
    except subprocess.CalledProcessError as e:
        click.echo(f"Error generating Tailscale certificates: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
