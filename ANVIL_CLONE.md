# Anvil app clone: direct changes for Uplink backend

The Anvil app is updated so that `get_ai_move` and `check_winner_server` are **only** provided by the Uplink backend (server.py on EC2/Lightsail). The Server Module no longer defines them.

**Clone URL:** `ssh://raghu.s%40utexas.edu@anvil.works:2222/ZYMBBJME4FOBYPVB.git`

---

## 1. Clone the app

From this project directory (or any folder where you want the repo):

```bash
git clone "ssh://raghu.s%40utexas.edu@anvil.works:2222/ZYMBBJME4FOBYPVB.git" anvil-connect4
```

**If clone fails** (e.g. "Host key verification failed" or "Permission denied"):

- Add your SSH key in Anvil: open the app in Anvil, go to **Settings → Git**, and add your public key.
- Ensure you can reach `anvil.works` on port 2222 from your machine.
- Then run the `git clone` again from a terminal where SSH to Anvil works.

---

## 2. Apply the Server Module edit

Replace the Anvil app’s Server Module with the Uplink-only version:

```bash
cp "ANVIL_ServerModule1_edited.py" "anvil-connect4/server_code/ServerModule1.py"
```

Or open `anvil-connect4/server_code/ServerModule1.py` in an editor and replace its contents with the contents of `ANVIL_ServerModule1_edited.py` in this project.

The edited module keeps the same imports and removes the definitions of `check_winner_server` and `get_ai_move` (they are provided by the Uplink when it is connected).

---

## 3. Commit and push

From the clone directory:

```bash
cd anvil-connect4
git add server_code/ServerModule1.py
git commit -m "Server: get_ai_move and check_winner_server provided by Uplink backend"
git push
```

**If push fails** (e.g. authentication or permissions):

- Fix SSH or Anvil Git access (Settings → Git), then run `git push` again from `anvil-connect4/`.
- The commit is already saved locally; pushing updates the app on Anvil.

---

## 4. Remaining steps for you

- **Push the clone to Anvil**  
  If the tool did not push (e.g. SSH/auth), run `git push` from the `anvil-connect4/` directory after fixing access.

- **Enable the Uplink**  
  In the Anvil app: **App → Settings → Uplink**. Ensure the Uplink is enabled for this app.

- **Run the backend**  
  Deploy and run `server.py` on EC2/Lightsail as described in **AWS_LIGHTSAIL_DEPLOY.md**. When the Uplink is connected, the app will use that backend for `get_ai_move` and `check_winner_server`.

No client code changes are required; the app keeps calling `get_ai_move(board)` and `check_winner_server(board, piece)`.
