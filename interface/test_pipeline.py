import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from pipeline import SyllabiPipeline

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
        
    def test_reformat_openlibracy_lccn_valid(self):
        lccn = 'ABC123'
        result = self.pipeline._reformat_openlibrary_lccn(lccn)
        expected = 'ABC123'
        self.assertEqual(result, expected)
    
    
        
        
        
    
if __name__ == '__main__':
    unittest.main()