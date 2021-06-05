"""General tests that check if pytest is working as expected"""

import pytest


def test_always_passes():
    assert True


def test_uppercase():
    assert "loud noises".upper() == "LOUD NOISES"


def test_reversed():
    assert list(reversed([1, 2, 3, 4])) == [4, 3, 2, 1]


def test_some_primes():
    def generator(num):
        for i in range(2, num):
            yield num % i == 0

    assert 37 in {
        num
        for num in range(1, 50)
        if num != 1 and not any(generator(num))
    }


@pytest.fixture(name="people_data")
def fixture_people_data():
    return [
        {
            "given_name": "Alfonsa",
            "family_name": "Ruiz",
            "title": "Senior Software Engineer",
        },
        {
            "given_name": "Sayid",
            "family_name": "Khan",
            "title": "Project Manager",
        },
    ]


def test_format_data_for_display(people_data):
    # Use data from fixture in test
    assert people_data[0]["given_name"] == "Alfonsa"


@pytest.mark.parametrize("strings", [
    "",
    "a",
    "Bob",
    "Never odd or even",
    "Do geese see God?",
])
def test_is_string(strings):
    assert isinstance(strings, str)
