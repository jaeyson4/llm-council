"""Interactive terminal tool to delete saved LLM Council conversations.

Run from the project root:

    python -m backend.delete_chat

It lists every stored conversation (newest first), lets you pick one, and then
requires TWO separate confirmations before the conversation is permanently
removed. Deletion only touches the JSON file in data/conversations/ -- nothing
is sent anywhere, and any Obsidian notes are left untouched.
"""

from . import storage


def _prompt(msg: str) -> str:
    """input() that treats Ctrl-C / Ctrl-D as 'cancel' instead of crashing."""
    try:
        return input(msg).strip()
    except (EOFError, KeyboardInterrupt):
        print()  # move to a fresh line after ^C / ^D
        return ""


def main() -> None:
    conversations = storage.list_conversations()
    if not conversations:
        print("No conversations found in data/conversations/. Nothing to delete.")
        return

    print("Saved conversations (newest first):\n")
    for i, c in enumerate(conversations, start=1):
        print(f"  {i:>2}. {c['title']}")
        print(f"      id: {c['id']}  |  {c['message_count']} message(s)  |  {c['created_at']}")
    print()

    choice = _prompt(
        f"Enter the number of the chat to delete (1-{len(conversations)}), or 'q' to quit: "
    )
    if not choice or choice.lower() in ("q", "quit", "exit"):
        print("Cancelled. Nothing was deleted.")
        return
    if not choice.isdigit() or not (1 <= int(choice) <= len(conversations)):
        print(f"'{choice}' is not a valid selection. Nothing was deleted.")
        return

    target = conversations[int(choice) - 1]
    title, conv_id = target["title"], target["id"]

    # --- Confirmation #1: y/N ---
    first = _prompt(
        f'\nDelete "{title}" ({conv_id})? This cannot be undone. [y/N]: '
    )
    if first.lower() not in ("y", "yes"):
        print("Cancelled. Nothing was deleted.")
        return

    # --- Confirmation #2: must type the word 'delete' ---
    second = _prompt("Are you REALLY sure? Type 'delete' to confirm: ")
    if second.lower() != "delete":
        print("Cancelled. Nothing was deleted.")
        return

    if storage.delete_conversation(conv_id):
        print(f'Deleted "{title}" ({conv_id}).')
    else:
        print(f"Could not find a file for {conv_id}; it may already be gone.")


if __name__ == "__main__":
    main()
