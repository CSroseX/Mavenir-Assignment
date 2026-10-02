from src.generation.nodes import _detect_banned_phrase


class TestBannedPhraseGateFalsePositives:
    """Legitimate 3GPP prose about retrieving subscription data must not
    trip the gate — the old bare-substring check on "retrieved"/"chunks"
    did exactly this."""

    def test_domain_use_of_retrieved_does_not_trigger(self):
        answer = "The UE retrieved the subscription data from the UDM during registration."
        assert _detect_banned_phrase(answer) is None

    def test_domain_use_of_chunk_free_text_does_not_trigger(self):
        answer = "The AMF manages mobility and registration for the UE across tracking areas."
        assert _detect_banned_phrase(answer) is None

    def test_word_the_chunks_is_still_flagged(self):
        # "the chunks" (plural, with article) is specific enough to the
        # retrieval system that it should still be caught.
        answer = "This isn't mentioned in the chunks I have."
        assert _detect_banned_phrase(answer) is not None


class TestBannedPhraseGateTruePositives:
    def test_provided_source_is_flagged(self):
        assert _detect_banned_phrase("Based on the provided source, the AMF handles mobility.") is not None

    def test_retrieved_information_is_flagged(self):
        assert _detect_banned_phrase("The retrieved information doesn't cover this case.") is not None

    def test_provided_specifications_is_flagged(self):
        assert _detect_banned_phrase("This isn't defined in the provided specifications.") is not None

    def test_case_insensitive(self):
        assert _detect_banned_phrase("PROVIDED SOURCE says otherwise.") is not None
