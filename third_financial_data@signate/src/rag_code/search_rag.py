import re
import os
import pymupdf4llm
from itertools import combinations

class PDFSearcher:
    def __init__(self, base_dir, context_words=50):
        """
        Initialize PDFSearcher with a base directory containing PDFs and/or markdown files
        Args:
            base_dir (str): Base directory containing company documents
            context_words (int): Number of context words for search results
        """
        self.base_dir = base_dir
        self.context_words = context_words
        self.markdown_text = None
        self.current_company = None
        self.company_to_file = {}
        self._scan_directory()

    def _scan_directory(self):
        """Scan directory for PDF and MD files and map them to company names"""
        for filename in os.listdir(self.base_dir):
            if filename.lower().endswith(('.pdf', '.md')):
                company_name = self._extract_company_name(filename)
                self.company_to_file[company_name] = os.path.join(self.base_dir, filename)

    def _extract_company_name(self, filename: str) -> str:
        """Extract company name from filename"""
        return filename.split('-')[0]

    def find_target_company(self, query: str) -> str:
        """
        Extract target company name from query
        Args:
            query (str): Search query
        Returns:
            str: Extracted company name or None
        """
        target_company = None
        max_length = 0

        for company in self.company_to_file.keys():
            if company in query and len(company) > max_length:
                target_company = company
                max_length = len(company)

        return target_company

    def load_document(self, company_name):
        """
        Load document for specified company, converting PDF to markdown if necessary
        Args:
            company_name (str): Name of the company
        Returns:
            bool: True if successful, False otherwise
        """
        if company_name not in self.company_to_file:
            print(f"No document found for company: {company_name}")
            return False

        self.current_company = company_name
        file_path = self.company_to_file[company_name]

        if file_path.lower().endswith('.md'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.markdown_text = f.read()
                print(f"Loaded markdown file for {company_name}")
                return True
            except Exception as e:
                print(f"Error loading markdown file: {e}")
                return False
        else:  # PDF file
            try:
                self.markdown_text = pymupdf4llm.to_markdown(file_path)
                print(f"Converted PDF to markdown for {company_name}")
                return True
            except Exception as e:
                print(f"Error converting PDF to markdown: {e}")
                return False

    def save_markdown(self, output_path=None):
        """
        Save current markdown text to file
        Args:
            output_path (str, optional): Output path. If None, uses company name
        """
        if self.markdown_text is None:
            print("No markdown text available.")
            return

        if output_path is None and self.current_company:
            output_path = os.path.join(self.base_dir, f"{self.current_company}.md")

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(self.markdown_text)
            print(f"Markdown saved to: {output_path}")
        except Exception as e:
            print(f"Error saving markdown: {e}")

    def search_keyword_combination(self, keywords, window_size=None):
        """
        Search for combinations of keywords within a specified window
        Args:
            keywords (list): List of keywords to search for
            window_size (int, optional): Maximum words between keywords
        Returns:
            dict: Dictionary of search results organized by keyword combinations
        """
        if self.markdown_text is None:
            print("No markdown text available. Load a document first.")
            return {}

        if window_size is None:
            window_size = self.context_words

        all_results = {}
        for r in range(1, len(keywords) + 1):
            for combo in combinations(keywords, r):
                combo_str = " AND ".join(combo)
                results = self._search_keyword_combination(combo, window_size)
                if results:
                    all_results[combo_str] = results

        return all_results

    def _search_keyword_combination(self, keywords, window_size):
        """
        Search for a specific combination of keywords
        Args:
            keywords (tuple): Tuple of keywords to search for
            window_size (int): Maximum words between keywords
        Returns:
            list: List of matches with context
        """
        text_lower = self.markdown_text.lower()
        matches = []

        first_keyword = keywords[0].lower()
        for match in re.finditer(first_keyword, text_lower):
            start_pos = match.start()

            words = self.markdown_text[start_pos:].split()
            window_text = " ".join(words[:window_size]).lower()

            all_found = True
            for keyword in keywords[1:]:
                if keyword.lower() not in window_text:
                    all_found = False
                    break

            if all_found:
                match_word_pos = len(self.markdown_text[:start_pos].split())
                context_start = max(0, match_word_pos - self.context_words)
                context_end = min(len(self.markdown_text.split()),
                                match_word_pos + self.context_words + 1)

                context = " ".join(self.markdown_text.split()[context_start:context_end])

                matches.append({
                    "keywords": keywords,
                    "context": context,
                    "position": start_pos,
                    "context_range": (context_start, context_end)
                })

        return matches

# Usage example:
if __name__ == "__main__":
    # Initialize searcher with directory containing company documents
    searcher = PDFSearcher("./company_docs")

    # Example query
    query = "ウエルシアホールディングスが掲げる2030のありたい姿はなんですか？"

    # Find target company from query
    company = searcher.find_target_company(query)
    if company:
        print(f"Found target company: {company}")

        # Load document for the company
        if searcher.load_document(company):
            # Search for keywords
            keywords = ["事業戦略", "成長"]
            results = searcher.search_keyword_combination(keywords)

            # Print results
            if results:
                for combo, matches in results.items():
                    print(f"\nResults for {combo}:")
                    for match in matches:
                        print(f"Context: {match['context']}")
