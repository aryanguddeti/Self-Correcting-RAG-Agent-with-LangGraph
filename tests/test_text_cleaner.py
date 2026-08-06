from src.modules.ingestion.preprocessor import DocumentPreprocessor, HeaderPropagator

preprocessor = DocumentPreprocessor()
propagator = HeaderPropagator()


# ── 1. clean_pdf_artifacts ────────────────────────────────────────────────────

def test_page_number_removal():
    text = "Some content\nPage 3 of 10\nMore content"
    result = preprocessor.clean_pdf_artifacts(text)
    assert "Page 3 of 10" not in result
    print("✅ Page number removed")


def test_dash_page_marker_removal():
    text = "Some content\n- 5 -\nMore content"
    result = preprocessor.clean_pdf_artifacts(text)
    assert "- 5 -" not in result
    print("✅ Dash page marker removed")


def test_markdown_image_removal():
    text = "See figure: ![diagram](https://example.com/img.png) for details."
    result = preprocessor.clean_pdf_artifacts(text)
    assert "![" not in result
    print("✅ Markdown image removed")


def test_html_tag_removal():
    text = "Hello <br> World <div class='x'>content</div>"
    result = preprocessor.clean_pdf_artifacts(text)
    assert "<br>" not in result and "<div" not in result
    print("✅ HTML tags removed")


def test_null_byte_removal():
    text = "Hello\x00World"
    result = preprocessor.clean_pdf_artifacts(text)
    assert "\x00" not in result
    print("✅ Null byte removed")


def test_excessive_blank_lines_collapsed():
    text = "Para one\n\n\n\nPara two"
    result = preprocessor.clean_pdf_artifacts(text)
    assert "\n\n\n" not in result
    print("✅ Excessive blank lines collapsed")


def test_table_preserved():
    table = "| Col1 | Col2 |\n|------|------|\n| A    | B    |"
    result = preprocessor.clean_pdf_artifacts(table)
    assert "|------|" in result
    print("✅ Table structure preserved")


# ── 2. normalize_text ─────────────────────────────────────────────────────────

def test_ligature_expansion():
    text = "The ﬁle has a ﬂaw and ﬀ issues."
    result = preprocessor.normalize_text(text)
    assert "fi" in result and "fl" in result and "ff" in result
    print("✅ Ligatures expanded")


def test_non_breaking_space_normalized():
    text = "Hello\xa0World"
    result = preprocessor.normalize_text(text)
    assert "\xa0" not in result
    print("✅ Non-breaking space normalized")


def test_zero_width_space_removed():
    text = "Hello\u200bWorld"
    result = preprocessor.normalize_text(text)
    assert "\u200b" not in result
    print("✅ Zero-width space removed")


def test_hyphenated_line_break_rejoined():
    text = "This is a pre-\nprocessing step."
    result = preprocessor.normalize_text(text)
    assert "preprocessing" in result
    print("✅ Hyphenated line break rejoined")


def test_multiple_spaces_collapsed():
    text = "Too    many     spaces"
    result = preprocessor.normalize_text(text)
    assert "  " not in result
    print("✅ Multiple spaces collapsed")


# ── 3. redact_pii ─────────────────────────────────────────────────────────────

def test_email_redacted():
    text = "Contact us at john.doe@example.com for support."
    result = preprocessor.redact_pii(text)
    assert "john.doe@example.com" not in result
    print("✅ Email redacted")


def test_phone_redacted():
    text = "Call us at 415-555-0198."
    result = preprocessor.redact_pii(text)
    assert "415-555-0198" not in result
    print("✅ Phone number redacted")


def test_person_name_redacted():
    text = "The policy holder is John Smith."
    result = preprocessor.redact_pii(text)
    assert "John Smith" not in result
    print("✅ Person name redacted")


# ── 4. HeaderPropagator ───────────────────────────────────────────────────────

def test_breadcrumb_prepended_to_body():
    md = "# Benefits\n## Dental\nCovers routine checkups."
    result = propagator.propagate(md)
    assert "Benefits > Dental" in result
    print("✅ Breadcrumb prepended to body paragraph")


def test_deeper_headers_cleared_on_level_up():
    md = "# Section A\n## Sub A\nText A\n# Section B\nText B"
    result = propagator.propagate(md)
    section_b_part = result.split("Section B")[1]
    assert "Sub A" not in section_b_part
    print("✅ Stale sub-header cleared on level-up")


def test_header_block_not_double_wrapped():
    md = "# Introduction\nSome intro text."
    result = propagator.propagate(md)
    assert result.count("**[Context:") <= 1
    print("✅ Header block not double-wrapped")


# ── Run all ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n── clean_pdf_artifacts ──")
    test_page_number_removal()
    test_dash_page_marker_removal()
    test_markdown_image_removal()
    test_html_tag_removal()
    test_null_byte_removal()
    test_excessive_blank_lines_collapsed()
    test_table_preserved()

    print("\n── normalize_text ──")
    test_ligature_expansion()
    test_non_breaking_space_normalized()
    test_zero_width_space_removed()
    test_hyphenated_line_break_rejoined()
    test_multiple_spaces_collapsed()

    print("\n── redact_pii ──")
    test_email_redacted()
    test_phone_redacted()
    test_person_name_redacted()

    print("\n── HeaderPropagator ──")
    test_breadcrumb_prepended_to_body()
    test_deeper_headers_cleared_on_level_up()
    test_header_block_not_double_wrapped()

    print("\n✅ All tests passed.")
