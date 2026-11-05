"""Single-platform entry point sharing the production action/end/interview protocol."""

if __package__:
    from . import run_parallel_simulation as engine
else:
    import run_parallel_simulation as engine


class TwitterSimulationRunner:
    """Compatibility entry for callers that await run(max_rounds=...)."""
    AVAILABLE_ACTIONS = engine.TWITTER_ACTIONS

    def __init__(self, config_path, wait_for_commands=True):
        self.config_path = config_path
        self.wait_for_commands = wait_for_commands

    async def run(self, max_rounds=None):
        argv = ["--config", self.config_path]
        if max_rounds is not None:
            argv.extend(["--max-rounds", str(max_rounds)])
        if not self.wait_for_commands:
            argv.append("--no-wait")
        return await engine.main(argv, platform="twitter")


async def main(argv=None):
    return await engine.main(argv, platform="twitter")


if __name__ == "__main__":
    engine.cli(platform="twitter")
