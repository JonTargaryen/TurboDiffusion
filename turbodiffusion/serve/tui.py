"""
TurboDiffusion TUI Server Mode

A fancy, user-friendly text-based interface for video generation.
Supports both T2V (text-to-video) and I2V (image-to-video) modes.
"""

import argparse
import os

from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .utils import RUNTIME_PARAMS, validate_args, format_config, set_runtime_param
from .pipeline import load_models, generate_t2v, generate_i2v

console = Console()

# Slash commands
COMMANDS = {
    "/help": "Show available commands",
    "/show": "Show current configuration",
    "/set": "Set a runtime parameter: /set <param> <value>",
    "/reset": "Reset runtime parameters to defaults",
    "/quit": "Exit the server",
}

# Style for prompt_toolkit
PROMPT_STYLE = Style.from_dict({
    "prompt": "#00aa00 bold",
    "command": "#ffaa00",
})


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for TUI server mode."""
    parser = argparse.ArgumentParser(
        description="TurboDiffusion TUI Server - Interactive video generation"
    )

    parser.add_argument("--mode", choices=["t2v", "i2v"], default="t2v",
                        help="Generation mode: t2v (text-to-video) or i2v (image-to-video)")

    # T2V model path
    parser.add_argument("--dit_path", type=str, default=None,
                        help="Path to DiT checkpoint (required for t2v mode)")

    # I2V model paths
    parser.add_argument("--high_noise_model_path", type=str, default=None,
                        help="Path to high-noise model (required for i2v mode)")
    parser.add_argument("--low_noise_model_path", type=str, default=None,
                        help="Path to low-noise model (required for i2v mode)")
    parser.add_argument("--boundary", type=float, default=0.9,
                        help="Timestep boundary for model switching (i2v only)")

    # Model configuration
    parser.add_argument("--model", choices=["Wan2.1-1.3B", "Wan2.1-14B", "Wan2.2-A14B"],
                        default=None, help="Model architecture (auto-detected from mode if not set)")
    parser.add_argument("--vae_path", type=str, default="checkpoints/Wan2.1_VAE.pth",
                        help="Path to the Wan2.1 VAE")
    parser.add_argument("--text_encoder_path", type=str,
                        default="checkpoints/models_t5_umt5-xxl-enc-bf16.pth",
                        help="Path to the umT5 text encoder")

    # Resolution
    parser.add_argument("--resolution", default=None, type=str,
                        help="Resolution (default: 480p for t2v, 720p for i2v)")
    parser.add_argument("--aspect_ratio", default="16:9", type=str,
                        help="Aspect ratio (width:height)")
    parser.add_argument("--adaptive_resolution", action="store_true",
                        help="Adapt resolution to input image aspect ratio (i2v only)")

    # Attention/quantization
    parser.add_argument("--attention_type", choices=["sla", "sagesla", "original"],
                        default="sagesla", help="Attention mechanism type")
    parser.add_argument("--sla_topk", type=float, default=0.1,
                        help="Top-k ratio for SLA/SageSLA attention")
    parser.add_argument("--quant_linear", action="store_true",
                        help="Use quantized linear layers")
    parser.add_argument("--default_norm", action="store_true",
                        help="Use default LayerNorm/RMSNorm (not optimized)")

    # Sampling options
    parser.add_argument("--ode", action="store_true",
                        help="Use ODE sampling (sharper but less robust, i2v only)")

    # Runtime-adjustable parameters
    parser.add_argument("--num_steps", type=int, choices=[1, 2, 3, 4], default=4,
                        help="Number of inference steps (1-4)")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of samples to generate")
    parser.add_argument("--num_frames", type=int, default=81,
                        help="Number of frames to generate")
    parser.add_argument("--sigma_max", type=float, default=None,
                        help="Initial sigma (default: 80 for t2v, 200 for i2v)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility")

    return parser.parse_args()


def print_header(args: argparse.Namespace):
    """Print fancy server header."""
    from rcm.datasets.utils import VIDEO_RES_SIZE_INFO
    w, h = VIDEO_RES_SIZE_INFO[args.resolution][args.aspect_ratio]
    mode_str = "[cyan]T2V[/cyan] (text-to-video)" if args.mode == "t2v" else "[magenta]I2V[/magenta] (image-to-video)"

    header = Text()
    header.append("TurboDiffusion TUI Server\n", style="bold blue")
    header.append(f"Mode: {mode_str}\n")
    header.append(f"Model: [green]{args.model}[/green] | ")
    header.append(f"Resolution: [yellow]{args.resolution}[/yellow] ({w}x{h}) | ")
    header.append(f"Steps: [yellow]{args.num_steps}[/yellow]")

    console.print(Panel(header, border_style="blue"))
    console.print("[dim]Type [bold]/help[/bold] for commands. Use [bold]\\\\[/bold] for newline in prompts.[/dim]\n")


def print_help():
    """Print help for slash commands."""
    table = Table(title="Commands", show_header=True, header_style="bold cyan")
    table.add_column("Command", style="yellow")
    table.add_column("Description")

    for cmd, desc in COMMANDS.items():
        table.add_row(cmd, desc)

    console.print(table)

    # Runtime params
    console.print("\n[bold cyan]Runtime Parameters[/bold cyan] (adjustable with /set):")
    for param, spec in RUNTIME_PARAMS.items():
        if "choices" in spec:
            console.print(f"  [yellow]{param}[/yellow]: {spec['choices']}")
        else:
            console.print(f"  [yellow]{param}[/yellow]: {spec['type'].__name__} (min: {spec.get('min', 'none')})")


def print_config(args: argparse.Namespace, defaults: dict):
    """Print current configuration."""
    console.print(format_config(args, defaults), markup=True)


def get_prompt_input(history: InMemoryHistory) -> str:
    """Get prompt from user with slash command completion."""
    completer = WordCompleter(list(COMMANDS.keys()), ignore_case=True)

    try:
        text = prompt(
            [("class:prompt", "prompt> ")],
            style=PROMPT_STYLE,
            completer=completer,
            history=history,
            multiline=False,
        )
        # Handle backslash for multiline: replace \\ with actual newline
        text = text.replace("\\\\", "\n")
        return text.strip()
    except (EOFError, KeyboardInterrupt):
        return None


def get_path_input(prompt_text: str, default: str = None, must_exist: bool = False) -> str:
    """Get file path from user."""
    default_hint = f" [{default}]" if default else ""
    try:
        text = prompt(
            [("class:prompt", f"{prompt_text}{default_hint}: ")],
            style=PROMPT_STYLE,
        )
        text = text.strip()

        if not text and default:
            return default

        if must_exist and text and not os.path.isfile(text):
            console.print(f"[red]Error: File not found: {text}[/red]")
            return None

        return text if text else None
    except (EOFError, KeyboardInterrupt):
        return None


def handle_command(cmd: str, args: argparse.Namespace, defaults: dict) -> bool:
    """Handle slash command. Returns False if should quit."""
    parts = cmd.strip().split()
    command = parts[0].lower()

    if command == "/quit":
        return False
    elif command == "/help":
        print_help()
    elif command == "/show":
        print_config(args, defaults)
    elif command == "/set":
        if len(parts) != 3:
            console.print("[red]Usage: /set <param> <value>[/red]")
        else:
            success, msg = set_runtime_param(args, parts[1], parts[2])
            if success:
                console.print(f"[green]{msg}[/green]")
            else:
                console.print(f"[red]Error: {msg}[/red]")
    elif command == "/reset":
        for param, default in defaults.items():
            setattr(args, param, default)
        console.print("[green]Runtime parameters reset to defaults.[/green]")
    else:
        console.print(f"[red]Unknown command: {command}[/red]")
        console.print("[dim]Type /help for available commands.[/dim]")

    return True


def run_tui(models: dict, args: argparse.Namespace):
    """Main TUI loop."""
    defaults = {param: getattr(args, param) for param in RUNTIME_PARAMS}
    last_output_path = "output/generated_video.mp4"
    last_image_path = None

    prompt_history = InMemoryHistory()

    print_header(args)

    while True:
        # Get prompt
        user_input = get_prompt_input(prompt_history)

        if user_input is None:
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue

        # Handle slash commands
        if user_input.startswith("/"):
            if not handle_command(user_input, args, defaults):
                console.print("[dim]Goodbye![/dim]")
                break
            continue

        prompt_text = user_input

        # For I2V mode, get image path
        image_path = None
        if args.mode == "i2v":
            image_path = get_path_input("image", last_image_path, must_exist=True)
            if image_path is None:
                console.print("[yellow]Cancelled.[/yellow]")
                continue
            last_image_path = image_path

        # Get output path
        output_path = get_path_input("output", last_output_path)
        if output_path is None:
            console.print("[yellow]Cancelled.[/yellow]")
            continue

        if not output_path.endswith(".mp4"):
            output_path += ".mp4"

        # Generate
        console.print()
        try:
            with console.status("[bold green]Generating video...", spinner="dots"):
                if args.mode == "t2v":
                    result_path = generate_t2v(models, args, prompt_text, output_path)
                else:
                    result_path = generate_i2v(models, args, prompt_text, image_path, output_path)

            console.print(f"[bold green]Done:[/bold green] {result_path}")
            last_output_path = result_path
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            import traceback
            traceback.print_exc()

        console.print()


def main(passed_args: argparse.Namespace = None):
    """Main entry point for TUI server."""
    args = passed_args if passed_args is not None else parse_arguments()

    validate_args(args)

    console.print("[dim]Loading models...[/dim]")
    models = load_models(args)

    try:
        run_tui(models, args)
    except KeyboardInterrupt:
        console.print("\n\n[dim]Interrupted. Goodbye![/dim]")


if __name__ == "__main__":
    main()
