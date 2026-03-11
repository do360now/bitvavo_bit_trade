
start_claude:
	ollama launch claude --model minimax-m2.5:cloud


run_tests:
	pytest -v --disable-warnings --maxfail=1 tests/

