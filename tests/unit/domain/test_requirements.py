from resume_shortlister.domain.requirements import MAX_REQUIREMENT_CHARS, extract_requirements
from tests.conftest import ANDROID_JD


def texts(requirements):
    return [r.text for r in requirements]


def test_bullets_become_requirements_and_headings_are_dropped():
    requirements = extract_requirements(ANDROID_JD)
    assert texts(requirements) == [
        "Android Developer (Kotlin)",
        "2+ years building Android apps with Kotlin",
        "Jetpack Compose, MVVM architecture and Coroutines",
        "Firebase Auth, Firestore and Cloud Messaging",
        "Publishing apps on the Google Play Store",
        "CI/CD with GitHub Actions",
    ]
    assert [r.id for r in requirements] == [1, 2, 3, 4, 5, 6]


def test_nice_to_have_items_are_optional_with_half_weight():
    requirements = extract_requirements(ANDROID_JD)
    optional = [r for r in requirements if r.optional]
    assert texts(optional) == ["CI/CD with GitHub Actions"]
    assert optional[0].weight == 0.5
    assert all(r.weight == 1.0 for r in requirements if not r.optional)


def test_benefits_and_about_sections_are_ignored():
    jd = "About us:\nWe are a fast-growing startup.\n\nSkills:\n- Python\n\nPerks\n- Free lunch"
    assert texts(extract_requirements(jd)) == ["Python"]


def test_inline_heading_switches_section():
    jd = "Must have: Kotlin and Compose\nNice to have: Flutter"
    requirements = extract_requirements(jd)
    assert [(r.text, r.optional) for r in requirements] == [
        ("Kotlin and Compose", False),
        ("Flutter", True),
    ]


def test_sentences_are_split_but_abbreviations_are_not():
    jd = "Strong Python skills, e.g. FastAPI or Django. Experience with AWS. B.Tech. preferred"
    assert texts(extract_requirements(jd)) == [
        "Strong Python skills, e.g. FastAPI or Django.",
        "Experience with AWS.",
        "B.Tech. preferred",
    ]


def test_boilerplate_sentences_and_duplicates_are_removed():
    jd = "Kotlin\nkotlin\nCompetitive salary and benefits.\nWe are an equal opportunity employer."
    assert texts(extract_requirements(jd)) == ["Kotlin"]


def test_single_line_description_falls_back_to_whole_text():
    assert texts(extract_requirements("app developer kotlin firebase")) == [
        "app developer kotlin firebase"
    ]


def test_cap_keeps_required_items_before_optional_ones():
    jd = "Nice to have:\n- Docker\n- Kubernetes\nRequirements:\n- Python\n- SQL"
    requirements = extract_requirements(jd, max_requirements=3)
    assert [(r.text, r.optional) for r in requirements] == [
        ("Docker", True),
        ("Python", False),
        ("SQL", False),
    ]


def test_long_requirements_are_truncated_on_a_word_boundary():
    jd = "Experience " + "with distributed systems " * 40
    (requirement,) = extract_requirements(jd)
    assert len(requirement.text) <= MAX_REQUIREMENT_CHARS
    assert requirement.text.endswith(("with", "distributed", "systems"))


def test_devanagari_sentence_terminator_splits_sentences():
    jd = "कोटलिन में अनुभव। फायरबेस का ज्ञान।"
    assert len(extract_requirements(jd)) == 2
