#!/usr/bin/env python3
"""
Unified Accelerate Configuration Script
Works on Windows, Linux, macOS, and RunPod
Automatically detects platform and configures accelerate accordingly
"""

import os
import sys
import platform
import shutil
import subprocess
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def run_cmd(cmd):
    """
    Execute a command using subprocess.
    """
    log.debug(f"Running command: {cmd}")
    try:
        if isinstance(cmd, str):
            # If cmd is a string, run it with shell=True
            process = subprocess.run(cmd, shell=True, check=True, env=os.environ, capture_output=True, text=True)
        else:
            # If cmd is a list, run it directly
            process = subprocess.run(cmd, check=True, env=os.environ, capture_output=True, text=True)

        log.debug(f"Command executed successfully: {cmd}")
        if process.stdout:
            log.debug(f"Stdout: {process.stdout.strip()}")
        if process.stderr:
            log.debug(f"Stderr: {process.stderr.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        log.error(f"Error occurred while running command: '{cmd}'. Exit code: {e.returncode}")
        if e.stdout:
            log.error(f"Stdout: {e.stdout.strip()}")
        if e.stderr:
            log.error(f"Stderr: {e.stderr.strip()}")
        log.error("Please check the command syntax, permissions, and ensure all required programs are installed and in PATH.")
        return False
    except FileNotFoundError:
        log.error(f"Command not found: {cmd}")
        return False

def detect_platform():
    """Detect the current platform/environment"""
    system = platform.system().lower()

    # Check for RunPod environment
    if os.path.exists("/workspace") and os.path.exists("/root/.cache"):
        return "runpod"

    # Check for other cloud environments or specific indicators
    if "RUNPOD" in os.environ or "/runpod" in os.getcwd():
        return "runpod"

    # Standard platform detection
    if system == "windows":
        return "windows"
    elif system == "linux":
        return "linux"
    elif system == "darwin":
        return "macos"
    else:
        return "unknown"

def get_accelerate_config_path():
    """Get the appropriate accelerate config path for the current environment"""
    # Special case for RunPod
    if detect_platform() == "runpod":
        return Path("/root/.cache/huggingface/accelerate/default_config.yaml")

    # Check environment variables in order of preference
    env_var_paths = []

    # HF_HOME takes precedence
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        env_var_paths.append(("HF_HOME", Path(hf_home) / "accelerate" / "default_config.yaml"))

    # XDG_CACHE_HOME for Linux/Unix
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        env_var_paths.append(("XDG_CACHE_HOME", Path(xdg_cache) / "huggingface" / "accelerate" / "default_config.yaml"))

    # Windows-specific paths
    if platform.system().lower() == "windows":
        local_appdata = os.environ.get("LOCALAPPDATA")
        if local_appdata:
            env_var_paths.append(("LOCALAPPDATA", Path(local_appdata) / "huggingface" / "accelerate" / "default_config.yaml"))

        userprofile = os.environ.get("USERPROFILE")
        if userprofile:
            env_var_paths.append(("USERPROFILE", Path(userprofile) / ".cache" / "huggingface" / "accelerate" / "default_config.yaml"))

    # Return the first path from environment variables if any exist
    if env_var_paths:
        return env_var_paths[0][1]

    # Default fallback paths
    if platform.system().lower() == "windows":
        userprofile = os.environ.get("USERPROFILE")
        if userprofile:
            return Path(userprofile) / ".cache" / "huggingface" / "accelerate" / "default_config.yaml"

    # Universal fallback
    return Path.home() / ".cache" / "huggingface" / "accelerate" / "default_config.yaml"

def get_source_config_file(platform_type):
    """Get the source config file based on platform"""
    script_dir = Path(__file__).parent

    if platform_type == "runpod":
        config_file = script_dir / "kohya_ss" / "config_files" / "accelerate" / "runpod.yaml"
    else:
        config_file = script_dir / "kohya_ss" / "config_files" / "accelerate" / "default_config.yaml"

    return config_file

def configure_accelerate_windows():
    """Configure accelerate on Windows"""
    log.info("Configuring accelerate for Windows...")
    success = run_cmd("accelerate config default")
    if success:
        log.info("Accelerate configured successfully on Windows")
    else:
        log.warning("Failed to configure accelerate automatically. Please run 'accelerate config' manually.")
    return success

def configure_accelerate_unix(platform_type):
    """Configure accelerate on Linux/macOS/RunPod"""
    log.info(f"Configuring accelerate for {platform_type}...")

    # Get source config file
    source_config = get_source_config_file(platform_type)
    if not source_config.exists():
        log.error(f"Source config file not found: {source_config}")
        log.warning("Falling back to manual accelerate configuration...")
        return run_cmd("accelerate config default")

    # Get target config path
    target_config = get_accelerate_config_path()

    log.debug(f"Source config: {source_config}")
    log.debug(f"Target config: {target_config}")

    try:
        # Create target directory if it doesn't exist
        target_config.parent.mkdir(parents=True, exist_ok=True)

        # Copy config file
        shutil.copyfile(source_config, target_config)
        log.info(f"Accelerate config copied successfully to: {target_config}")
        return True

    except Exception as e:
        log.error(f"Failed to copy accelerate config: {e}")
        log.warning("Falling back to manual accelerate configuration...")
        return run_cmd("accelerate config default")

def main():
    """Main function to configure accelerate"""
    log.info("Starting unified accelerate configuration...")

    # Detect platform
    platform_type = detect_platform()
    log.info(f"Detected platform: {platform_type}")

    # Configure based on platform
    if platform_type == "windows":
        success = configure_accelerate_windows()
    else:
        success = configure_accelerate_unix(platform_type)

    if success:
        log.info("Accelerate configuration completed successfully!")
    else:
        log.error("Accelerate configuration failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
