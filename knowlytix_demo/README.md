# knowlytix-demo

Support code and synthetic fixtures for the public KAL + GMS Colab demo.

The wheel contains:

- the six notebook helper modules
- the seven synthetic finance and governance fixtures
- the exact `knowlytix.harness.preflight` module from Harness 1.0.2.post1
- compatibility modules preserving the notebook's existing top-level imports
- resource helpers for deterministic fixture materialization

Installing this wheel does not provide a KnowlytiX license, an LLM API key, or
a Neo4j server. Those runtime services remain external requirements.
