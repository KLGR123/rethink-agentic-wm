"""The webshop server, agent loop, prompts, data preparation and training entry points.

Import agent_loop explicitly to register the training agent; data/server imports
do not load torch, Ray or the trainer.
"""
