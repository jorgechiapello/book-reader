import unittest
from text_extractors import RuleBasedTextExtractor, ReaderInterface, Chapter

class MockReader(ReaderInterface):
    def __init__(self, pages):
        self.pages = pages
    
    def extract_pages(self):
        return self.pages

class TestTextExtractors(unittest.TestCase):
    def test_should_split_text_when_line_break_followed_by_capital_letter(self):
        mock_pages = [
            'The £1,000,000 Bank-Note\nMark Twain\nWhen I was twenty-seven years old, I was a mining-broker’s clerk in San Francisco, and an \nexpert in all the details of stock traffic. I was alone in the world, and had nothing to depend \nupon but my wits and a clean reputation; but these were setting my feet in the road to eventual \nfortune, and I was content with the prospect.\nYou will remember that the Bank of England once issued two notes of a million pounds each, to ','be used for a special purpose connected with some public transaction with a foreign country. \nFor some reason or other only one of these had been used and cancelled; the other still lay in the \nvaults of the Bank.'
        ]
        reader = MockReader(mock_pages)
        extractor = RuleBasedTextExtractor(reader)
        chapters = extractor.extract_chapters()
        
        self.assertEqual(len(chapters), 1)
        self.assertEqual(chapters[0].title, '001_chapter 1')
        self.assertEqual(len(chapters[0].segments), 5)
        expected_segments = [
            'The £1,000,000 Bank-Note',
            'Mark Twain',
            'When I was twenty-seven years old, I was a mining-broker’s clerk in San Francisco, and an expert in all the details of stock traffic. I was alone in the world, and had nothing to depend upon but my wits and a clean reputation; but these were setting my feet in the road to eventual fortune, and I was content with the prospect.',
            'You will remember that the Bank of England once issued two notes of a million pounds each, to be used for a special purpose connected with some public transaction with a foreign country.', 
            'For some reason or other only one of these had been used and cancelled; the other still lay in the vaults of the Bank.'
        ]
        self.assertEqual(chapters[0].segments, expected_segments)

    def test_should_split_chapters_when_chapter_title_is_followed_by_text_on_new_line(self):
        mock_pages = [
            'The £1,000,000 Bank-Note\nMark Twain\nSome text.\nChapter 2\nSome more text'
        ]
        reader = MockReader(mock_pages)
        extractor = RuleBasedTextExtractor(reader)
        chapters = extractor.extract_chapters()
        
        self.assertEqual(len(chapters), 2)
        self.assertEqual(chapters[0].title, '001_chapter 1')
        self.assertEqual(len(chapters[0].segments), 3)
        expected_segments = [
            'The £1,000,000 Bank-Note',
            'Mark Twain',
            'Some text.'
        ]
        self.assertEqual(chapters[0].segments, expected_segments)
        self.assertEqual(chapters[1].title, '002_chapter 2')
        self.assertEqual(len(chapters[1].segments), 2)
        expected_segments = [
            'Chapter 2',
            'Some more text'
        ]
        self.assertEqual(chapters[1].segments, expected_segments)
if __name__ == "__main__":
    unittest.main()
