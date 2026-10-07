"""
Tests for the ticket and topic extractor module.
"""

import pytest
from src.extractor import extract_ticket_id, detect_topic


# ──────────────────────────────────────────────
# Ticket Extraction Tests
# ──────────────────────────────────────────────


class TestTicketExtraction:
    """Tests for extract_ticket_id()."""

    def test_simple_ticket(self):
        assert extract_ticket_id("My ticket is IT-9921") == "IT-9921"

    def test_ticket_at_start(self):
        assert extract_ticket_id("IT-1234 is my ticket") == "IT-1234"

    def test_ticket_at_end(self):
        assert extract_ticket_id("Can you check IT-5678?") == "IT-5678"

    def test_ticket_in_sentence(self):
        assert extract_ticket_id("I want an update for ticket IT-4321.") == "IT-4321"

    def test_ticket_with_surrounding_text(self):
        assert extract_ticket_id("Please look at IT-8812 for me") == "IT-8812"

    def test_no_ticket(self):
        assert extract_ticket_id("I need help with WiFi") is None

    def test_empty_string(self):
        assert extract_ticket_id("") is None

    def test_partial_ticket_too_few_digits(self):
        """IT- followed by fewer than 4 digits should not match."""
        assert extract_ticket_id("My ticket is IT-123") is None

    def test_partial_ticket_too_many_digits(self):
        """IT- followed by more than 4 digits should not match the full string."""
        # IT-12345 — the regex matches IT-1234 (first 4 digits), but
        # word boundary after the 4th digit won't match if digit 5 follows.
        assert extract_ticket_id("IT-12345") is None

    def test_not_a_ticket_pattern(self):
        assert extract_ticket_id("ADMIT-1234 is not a ticket") is None

    def test_case_sensitive(self):
        """Only uppercase IT- should match."""
        assert extract_ticket_id("my ticket is it-9921") is None

    def test_multiple_tickets_returns_last(self):
        """When multiple tickets exist, the last one should be returned."""
        result = extract_ticket_id(
            "My old ticket was IT-1111, but my new ticket is IT-2222."
        )
        assert result == "IT-2222"

    def test_multiple_tickets_three(self):
        result = extract_ticket_id("IT-1111 IT-2222 IT-3333")
        assert result == "IT-3333"

    def test_ticket_with_newlines(self):
        assert extract_ticket_id("Hello\nMy ticket is IT-7777\nThanks") == "IT-7777"

    def test_invalid_characters_in_id(self):
        assert extract_ticket_id("IT-ABCD") is None
        assert extract_ticket_id("IT-12AB") is None


# ──────────────────────────────────────────────
# Topic Detection Tests
# ──────────────────────────────────────────────


class TestTopicDetection:
    """Tests for detect_topic()."""

    def test_wifi_topic(self):
        assert detect_topic("I cannot connect to campus WiFi") == "wifi_issue"

    def test_wifi_topic_wireless(self):
        assert detect_topic("My wireless connection dropped") == "wifi_issue"

    def test_wifi_no_internet(self):
        assert detect_topic("Connected without internet on my laptop") == "wifi_issue"

    def test_ticket_topic(self):
        assert detect_topic("Where is my ticket status?") == "ticket_inquiry"

    def test_ticket_topic_with_id(self):
        assert detect_topic("My ticket is IT-1234") == "ticket_inquiry"

    def test_printing_topic(self):
        assert detect_topic("My printer is not working") == "printing_issue"

    def test_printing_topic_paper_jam(self):
        assert detect_topic("There's a paper jam in the lab printer") == "printing_issue"

    def test_password_topic(self):
        assert detect_topic("I forgot my university password") == "password_issue"

    def test_password_topic_reset(self):
        assert detect_topic("How do I reset password?") == "password_issue"

    def test_password_topic_locked(self):
        assert detect_topic("I am locked out of my account") == "password_issue"

    def test_software_topic(self):
        assert detect_topic("How do I install MATLAB?") == "software_issue"

    def test_software_topic_crash(self):
        assert detect_topic("My application keeps crashing") == "software_issue"

    def test_network_topic(self):
        assert detect_topic("The VPN is not connecting") == "network_issue"

    def test_network_topic_ethernet(self):
        assert detect_topic("My ethernet cable doesn't work") == "network_issue"

    def test_general_topic(self):
        assert detect_topic("Hi, I need help.") == "general"

    def test_general_topic_ambiguous(self):
        assert detect_topic("Hello, can you assist me?") == "general"

    def test_empty_string(self):
        assert detect_topic("") == "general"

    def test_case_insensitive(self):
        assert detect_topic("MY WIFI IS BROKEN") == "wifi_issue"
