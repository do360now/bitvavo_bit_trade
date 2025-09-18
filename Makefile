
lint_code:
	black bitcoin_bot/
	ruff check bitcoin_bot/ --fix
	pylint bitcoin_bot/
	isort bitcoin_bot/