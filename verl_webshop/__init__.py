# verl_webshop: WebShop environment integration for verl GRPO training.
# Import the agent loop so @register("webshop_agent") fires on import.

from verl_webshop.agent_loop import WebShopAgentLoop

__all__ = ["WebShopAgentLoop"]
