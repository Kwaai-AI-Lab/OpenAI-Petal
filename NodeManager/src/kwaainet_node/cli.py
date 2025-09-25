"""
Command-line interface for KwaaiNet NodeManager.
"""

import asyncio
import click
import logging
import sys
from pathlib import Path

from .core.config import ConfigManager
from .core.node_manager import NodeManager


@click.group()
@click.option('--config', '-c', 'config_path',
              help='Path to configuration file')
@click.option('--verbose', '-v', is_flag=True,
              help='Enable verbose logging')
@click.pass_context
def cli(ctx, config_path, verbose):
    """KwaaiNet NodeManager - Dynamic Process Orchestrator"""

    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Store context
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config_path
    ctx.obj['verbose'] = verbose


@cli.command()
@click.option('--daemon', '-d', is_flag=True,
              help='Run in daemon mode')
@click.pass_context
def start(ctx, daemon):
    """Start the NodeManager"""

    if daemon:
        click.echo("Daemon mode not yet implemented in MVP")
        sys.exit(1)

    try:
        # Load configuration
        config_manager = ConfigManager(ctx.obj.get('config_path'))
        config = config_manager.load_config()

        click.echo(f"Starting KwaaiNet NodeManager (Node ID: {config.node_id})")
        click.echo(f"Listening on port: {config.listen_port}")

        # Create and start node manager
        node_manager = NodeManager(config)

        # Run the node manager
        asyncio.run(node_manager.run())

    except KeyboardInterrupt:
        click.echo("\nShutdown requested by user")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def stop(ctx):
    """Stop the NodeManager (daemon mode)"""
    click.echo("Stop command not yet implemented (requires daemon mode)")


@cli.command()
@click.pass_context
def status(ctx):
    """Show NodeManager status"""
    click.echo("Status command not yet implemented")


@cli.command()
@click.pass_context
def config(ctx):
    """Show current configuration"""
    try:
        config_manager = ConfigManager(ctx.obj.get('config_path'))
        config = config_manager.load_config()

        click.echo(f"Configuration file: {config_manager.config_path}")
        click.echo(f"Node ID: {config.node_id}")
        click.echo(f"Listen Port: {config.listen_port}")
        click.echo(f"Default Memory Limit: {config.resource_schedule.default_limits.max_memory_gb:.1f} GB")
        click.echo(f"Default Model Limit: {config.resource_schedule.default_limits.max_models}")
        click.echo(f"Scheduled Limits: {len(config.resource_schedule.scheduled_limits)}")

        if config.resource_schedule.scheduled_limits:
            click.echo("\nScheduled Resource Limits:")
            for scheduled in config.resource_schedule.scheduled_limits:
                click.echo(f"  - {scheduled.name}: {scheduled.cron_expression} "
                          f"(Memory: {scheduled.limits.max_memory_gb:.1f} GB, "
                          f"Models: {scheduled.limits.max_models})")

    except Exception as e:
        click.echo(f"Error loading configuration: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--model', '-m', required=True,
              help='Model name to load')
@click.option('--blocks', '-b', required=True,
              help='Block range (e.g., "5:9" or "5,6,7,8")')
@click.pass_context
def load_model(ctx, model, blocks):
    """Load a specific model"""
    click.echo(f"Load model command not yet implemented")
    click.echo(f"Would load: {model} with blocks {blocks}")


@cli.command()
@click.option('--model', '-m', required=True,
              help='Model name to unload')
@click.pass_context
def unload_model(ctx, model):
    """Unload a specific model"""
    click.echo(f"Unload model command not yet implemented")
    click.echo(f"Would unload: {model}")


@cli.command()
@click.pass_context
def list_models(ctx):
    """List currently loaded models"""
    click.echo("List models command not yet implemented")


def main():
    """Main entry point for the CLI"""
    cli()


if __name__ == '__main__':
    main()