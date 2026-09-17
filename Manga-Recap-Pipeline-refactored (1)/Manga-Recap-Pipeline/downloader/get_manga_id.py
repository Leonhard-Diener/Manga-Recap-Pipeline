import requests
import config


def get_manga_id(title):
    response = requests.get(
        config.BASE_URL,
        params={
            "title": title,
            "order[relevance]": "desc",
            "limit": 10,
        },
        timeout=10,
    )
    response.raise_for_status()

    manga_results = response.json()["data"]

    if not manga_results:
        raise ValueError(f"No manga found for title '{title}'")

    print(f"\nSearch results for '{title}':\n")

    for i, manga in enumerate(manga_results, start=1):
        titles = manga["attributes"].get("title", {})

        display_title = (
            titles.get("en")
            or next(iter(titles.values()), "Unknown title")
        )

        print(f"{i}. {display_title}")

    while True:
        try:
            choice = int(
                input(f"\nWhich manga do you want? [1-{len(manga_results)}]: ")
            )

            if 1 <= choice <= len(manga_results):
                break

            print(
                f"Please enter a number between 1 and {len(manga_results)}."
            )

        except ValueError:
            print("Please enter a valid number.")

    manga = manga_results[choice - 1]

    manga_id = manga["id"]

    print(f"\nSelected: {manga['attributes']['title']}")

    return manga_id