# Using Claude/Copilot in this Repository

This project is built with support for AI-assisted development using GitHub Copilot (Raptor mini) and Claude models.  The work is organized around a trio of internal agents that help research, implement, and review changes.  Following the conventions below will make contributions smoother for both human developers and AI assistants.

## Repository Overview

- **Core modules**: `main.py`, `bitvavo_api.py`, `order_manager.py`, `trade_executor.py`, `data_manager.py`, `performance_tracker.py`, `indicators.py`, `circuit_breaker.py`, etc.  These implement the asynchronous Bitcoin trading bot logic.
- **Configuration and state**: `.env.template`, `bot_state.json`, `price_history.json`, and related JSON files hold run-time state and historical data.
- **Tests**: See `tests/` for unit and integration tests.  Running `python3 test_suite.py` and `python3 test_integration.py` exercises the code.
- **Documentation**: A set of `docs/` Markdown files describe analysis, refactoring summaries, and deployment notes.
- **Agents**: The `.github/agents/` directory contains agent definitions for automated workflows (`research.agent.md`, `implementer.agent.md`, `reviewer.agent.md`, and `feature-builder.agent.md`).  These files instruct Claude/Copilot how to behave when operating in this repo.

## Guidelines for AI Assistants

1. **Start with Researcher**
   - When first exploring or answering a high‑level question, use the `Researcher` agent and read-only tools (`search`, `fetch`, `usages`, `codebase`).
   - Gather information about module responsibilities, naming conventions, and existing patterns.
   - Summarize findings and hand off to the `Implementer` when a change is required.

2. **Implement Changes Carefully**
   - Use the `Implementer` agent to create or modify code/files.  Its toolset (`editFiles`, `terminal`, `search`) is sufficient for logical implementation.
   - Follow existing coding patterns: asynchronous functions use `async/await`, error handling uses the circuit breaker, and logging leverages `logger_config`.
   - Add or update tests in `tests/` alongside any functional change.
   - Run the code quality analyzer (`python3 code_quality_analyzer.py`) before committing.

3. **Request Review with Reviewer**
   - After implementation, invoke the `Reviewer` agent for a security and style review.
   - The reviewer should check for vulnerabilities, ensure best practices, and verify adequate test coverage.
   - Address any reviewer feedback before pushing changes.

4. **Branching & Commits**
   - Create feature branches prefixed with `feature/` or `fix/` as appropriate.
   - Write clear commit messages (e.g., `Add CLAUDE.md with AI usage guidance`).
   - Stage only relevant changes; avoid committing sensitive files or environment data.

5. **Model Selection & Conversations**
   - When prompted for a model, prefer **Raptor mini (Preview)** for Copilot tasks.
   - Use Claude Sonnet 4.5 (copilot) or GPT‑5.2 (copilot) as fallbacks per `.github/agents` configs.
   - Keep responses concise and follow the repository’s final answer formatting instructions (headings, bullet lists, code formatting).

6. **Testing & Quality**
   - Execute unit and integration tests after making changes.
   - Ensure no syntax errors via `python -m py_compile` or the Pylance analysis.
   - Maintain high code readability; use type hints where appropriate.

7. **Documentation**
   - Update README or relevant docs when adding features or changing behavior.
   - Place new explanatory files under `docs/` if they are substantial.

8. **Security & Secrets**
   - Never hard-code API keys or secrets; use the `.env` mechanism described in `README.md`.
   - Ensure logging obscures sensitive information.

By following this guidance, both human contributors and AI agents can collaborate effectively and maintain the high standard of this trading bot project.
