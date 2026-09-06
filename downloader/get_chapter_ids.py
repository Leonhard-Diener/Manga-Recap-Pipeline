import requests
import config


def get_chapter_ids(manga_id, language="en"):
    chapters = []

    offset = 0
    limit = 100

    while True:
        response = requests.get(
            f"{config.BASE_URL}/{manga_id}/feed",
            params={
                "translatedLanguage[]": language,
                "order[chapter]": "asc",
                "limit": limit,
                "offset": offset,
            },
            timeout=10,
        )
        response.raise_for_status()

        data = response.json()
        batch = data["data"]

        chapters.extend(batch)

        if not batch:
            break

        if offset + len(batch) >= data["total"]:
            break

        offset += len(batch)

    if not chapters:
        raise ValueError(
            f"No '{language}' chapters found for manga '{manga_id}'"
        )

    # Doppelte Kapitel entfernen
    unique_chapters = {}

    for chapter in chapters:
        chapter_number = chapter["attributes"].get("chapter")

        # Kapitel ohne Nummer nicht deduplizieren
        if chapter_number is None:
            unique_chapters[chapter["id"]] = chapter
            continue

        # Ersten Upload dieser Kapitelnummer behalten
        if chapter_number not in unique_chapters:
            unique_chapters[chapter_number] = chapter

    return [
        chapter["id"]
        for chapter in unique_chapters.values()
    ]