"""Ask one question from the command line and compare the three approaches.

    python ask.py "How many vacation days do I get?"
"""
import sys

from dotenv import load_dotenv

from assistant import build_all

load_dotenv()

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "How many vacation days do I get?"
    for key, approach in build_all().items():
        a = approach.ask(q)
        print(f"\n== {a.approach} ==")
        print(f"Answer : {a.answer}")
        print(f"Policy : {a.policy_title or '-'}")
        print(f"Time   : {a.time_ms:.0f} ms   Tokens: {a.total_tokens} "
              f"(in {a.input_tokens}, out {a.output_tokens})")
