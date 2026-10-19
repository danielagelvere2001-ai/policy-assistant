# Company Policy Assistant

A bot that answers company policy questions and names the relevant policy. It uses a database of 98 policies (`data/company_policies.csv`) and compares three approaches:

| Approach | How it works |
|---|---|
| **Rules-based search** | Keyword matching with stop words, light stemming, a hand-written synonym list and IDF weighting. It returns the best policy word for word and uses no LLM. |
| **LLM without vector index** | All 98 policies are pasted into the Gemini prompt for every question. |
| **LLM with vector index** | Policies are embedded once with Gemini embeddings and stored in a local index (`index/policy_index.npz`). Each question retrieves the top 3 policies by cosine similarity, and only those go to Gemini. |

The two LLM approaches share the same system prompt and JSON output schema, so the only difference is how the context is chosen.

- **Comparison website:** `docs/index.html` (served by GitHub Pages). It is generated from `results/results.json`.
- **Result (9 questions, gemini-3.6-flash):** rules-based search found the right policy 4/9 times and gave 5 unsupported answers; both LLM approaches found 9/9 with 0 unsupported answers. The vector-index approach used about 9x fewer tokens (≈320 vs ≈2,850 per question) and was faster (median 2.0 s vs 12.6 s). The Slack bot uses the vector-index approach.
- **Slack bot:** `slack_bot/app.py` runs locally in Socket Mode, so it needs no server or public URL.

## Project layout

```
assistant/        the three approaches + Gemini REST client
evaluation/       30 test questions and the evaluation runner (with LLM judge)
site_builder/     builds docs/index.html from the results; discussion.md = the two paragraphs
slack_bot/        Slack bot + app manifest
tests/            offline tests (no API key needed)
ask.py            ask one question from the terminal and compare the three approaches
```

## 1. Set up (Windows PowerShell)

```powershell
cd "path\to\policy-assistant"
python -m venv .venv
.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
copy .env.example .env
notepad .env                        # paste your GEMINI_API_KEY (https://aistudio.google.com/apikey)
python tests\test_offline.py        # should print "passed" 5 times
python ask.py "How many vacation days do I get?"
```

## 2. Run the comparison and build the website

```powershell
python -m evaluation.run_eval       # about 10-15 minutes on the free tier and resumable (Ctrl+C, then run again)
python -m site_builder.build_site   # writes docs\index.html
start docs\index.html               # preview it
```

`run_eval` pauses 15 s between questions so it stays under the free-tier rate limit (`--delay 0` if you have a paid key). Time spent waiting on rate-limit retries is subtracted from the measured response times.

The published run used 9 of the 30 questions (`--ids 1,5,6,12,21,23,26,27,30 --no-judge`) because the Gemini free tier allows only about 20 requests per model per day. The groundedness verdicts for that run were added by a reviewer afterwards, using the same rules as the automatic judge. With a paid key, run all 30 questions with the automatic judge.

What is measured per answer: the answer text, the policy it named, response time, tokens (from the Gemini API), whether the policy is correct, and a groundedness verdict. For that verdict, a Gemini judge sees the whole policy database and labels each answer *Supported*, *Unsupported* (it states something the database doesn't say, or presents an unrelated policy as the answer) or *No answer*. The 30 questions include 5 that no policy covers and 5 where the policy exists but lacks the detail asked about. Those 10 are the main tests of whether an approach makes things up.

## 3. Publish on GitHub Pages

1. Create a new **public** repository on GitHub, e.g. `policy-assistant`.
2. Push this folder (the `.env` file is git-ignored, so your keys stay local):
   ```powershell
   git init
   git add .
   git commit -m "Company policy assistant"
   git branch -M main
   git remote add origin https://github.com/<your-username>/policy-assistant.git
   git push -u origin main
   ```
3. On GitHub, go to **Settings → Pages → Build and deployment**. Choose *Deploy from a branch*, then branch `main`, folder `/docs`, and click **Save**.
4. After about a minute the site is live at `https://<your-username>.github.io/policy-assistant/`.

## 4. Slack bot setup

1. **Workspace:** go to <https://slack.com/get-started#/createnew> and create a new workspace whose name identifies you, e.g. **"Daniela – Policy Assistant"**.
2. **Channel:** create a channel such as `#policy-bot-testing`. Put your name in its description too, e.g. *"Daniela's testing channel for the policy bot"*.
3. **Invite the teacher:** in the channel, click the channel name → **Members → Add people**, enter `raz@sdu.dk` and send the invitation. If you're asked, invite them to the workspace first.
4. **Create the app:** go to <https://api.slack.com/apps> → **Create New App → From an app manifest**. Pick your new workspace, choose YAML, paste the contents of `slack_bot/manifest.yaml`, and create the app.
5. **App token (Socket Mode):** go to **Basic Information → App-Level Tokens → Generate Token and Scopes**, add scope `connections:write`, generate it, and copy the `xapp-...` token into `.env` as `SLACK_APP_TOKEN`.
6. **Bot token:** go to **Install App → Install to Workspace → Allow**, then copy the *Bot User OAuth Token* (`xoxb-...`) into `.env` as `SLACK_BOT_TOKEN`.
7. **Run it:**
   ```powershell
   python -m slack_bot.app
   ```
8. **Add the bot to the channel:** in `#policy-bot-testing` type `/invite @Policy Bot`.
9. **Test it and take the screenshot:**
   `@Policy Bot How many days a week can I work from home?`
   The bot replies in a thread with the answer, the relevant policy (title, category, owning department and the policy text), the time taken and the tokens used. `/policy <question>` and direct messages also work.
