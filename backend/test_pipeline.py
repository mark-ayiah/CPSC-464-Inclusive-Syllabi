import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from backend.pipeline import SyllabiPipeline

class TestPipeline(unittest.TestCase):
    
    pipeline = SyllabiPipeline()
    
    def test_split_lcc_valid(self):
        lcc = 'HV-1568.00000000.B376 2016'
        result = self.pipeline._split_lcc(lcc)
        expected = ('HV', 1568.0)
        self.assertEqual(result, expected)
        
    def test_lookup_meaning_valid(self):
        split_lcc = ('HV', 1568.0)
        result = self.pipeline._lookup_meaning(split_lcc)
        expected = ['Special classes', 'People with disabilities', 'Protection, assistance and relief', 'Social pathology.  Social and public welfare.']
        self.assertEqual(result, expected)
        
    @patch('requests.get')
    def test_searchby_lccn_valid(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {'docs': [{'title': 'test', 'lcc': ['ABC123']}]}
        mock_get.return_value = mock_response
        
        result = self.pipeline._searchby_lccn('ABC123\*', 'title, lcc', 1)
        expected = [{'title': 'test', 'lcc': ['ABC123']}]
        self.assertTrue(len(result) > 0)
        self.assertEqual(result, expected)
    
    @patch('requests.get') 
    def test_searchby_isbn_valid(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {'docs': [{"lcc": ['ABC123']}]}
        mock_get.return_value = mock_response
        
        result = self.pipeline._searchby_isbn('1234', 'lcc', 1)
        expected = 'ABC123'
        self.assertTrue(len(result) > 0)
        self.assertEqual(result, expected)
        
    @patch('pipeline.SyllabiPipeline._searchby_isbn')
    @patch('pipeline.SyllabiPipeline._split_lcc')
    def test_get_lccn_for_syllabus_valid(self, mock_split_lcc, mock_searchby_isbn):
        
        # expect three calls to split_lcc and searchby_isbn, one for each row in the test_syllabus dataframe
        mock_split_lcc.side_effect = [('ABC', 123.0), ('DEF', 456.0), ('GHI', 789.0)]
        mock_searchby_isbn.side_effect = ['ABC123', 'DEF456', 'GHI789']
        
        test_syllabus = pd.DataFrame({
            'isbn': ['123', '456', '789'], 
            'book_title': ['test1', 'test2', 'test3'], 
            'author': ['author1', 'author2', 'author3']
            })
        
        result = self.pipeline._get_lccn_for_syllabus(test_syllabus)
        
        expected = [('ABC', 123.0), ('DEF', 456.0), ('GHI', 789.0)]
        
        self.assertEqual(result, expected)
    
    def test_get_all_parents_valid(self):
        pass
    
    def test_find_most_recent_common_parent(self):
        pass
        
        
        
    
    
        
        
        
    
if __name__ == '__main__':
    unittest.main()