import json
import unittest
import re

from numpy import nan, dtype

from pandas_schema import Column, Schema
from pandas_schema.validation import _BaseValidation
from pandas_schema.validation import *
from pandas_schema import ValidationWarning


class ValidationTestBase(unittest.TestCase):
    def seriesEquality(self, s1: pd.Series, s2: pd.Series, msg: str = None):
        if not s1.equals(s2):
            raise self.failureException(msg)

    def validate_and_compare(self, series: list, expected_result: bool, msg: str = None, series_dtype: object = None):
        """
        Checks that every element in the provided series is equal to `expected_result` after validation
        :param series_dtype: Explicity specifies the dtype for the generated Series
        :param series: The series to check
        :param expected_result: Whether the elements in this series should pass the validation
        :param msg: The message to display if this test fails
        """

        # Check that self.validator is correct
        if not self.validator or not isinstance(self.validator, _BaseValidation):
            raise ValueError('The class must have the validator field set to an instance of a Validation subclass')

        # Ensure we're comparing series correctly
        self.addTypeEqualityFunc(pd.Series, self.seriesEquality)

        # Convert the input list to a series and validate it
        results = self.validator.validate(pd.Series(series, dtype=series_dtype))

        # Now find any items where their validation does not correspond to the expected_result
        for item, result in zip(series, results):
            with self.subTest(value=item):
                self.assertEqual(result, expected_result, msg)


class CustomSeries(ValidationTestBase):
    """
    Tests the CustomSeriesValidation
    """

    def setUp(self):
        self.validator = CustomSeriesValidation(lambda s: ~s.str.contains('fail'), 'contained the word fail')

    def test_valid_inputs(self):
        self.validate_and_compare(['good', 'success'], True, 'did not accept valid inputs')

    def test_invalid_inputs(self):
        self.validate_and_compare(['fail', 'failure'], False, 'accepted invalid inputs')


class CustomElement(ValidationTestBase):
    """
    Tests the CustomElementValidation
    """

    def setUp(self):
        self.validator = CustomElementValidation(lambda s: s.startswith('_start_'), "Didn't begin with '_start_'")

    def test_valid_inputs(self):
        self.validate_and_compare(['_start_sdiyhsd', '_start_234fpwunxc\n'], True, 'did not accept valid inputs')

    def test_invalid_inputs(self):
        self.validate_and_compare(['fail', '324wfp9ni'], False, 'accepted invalid inputs')


class LeadingWhitespace(ValidationTestBase):
    """
    Tests the LeadingWhitespaceValidation
    """

    def setUp(self):
        self.validator = LeadingWhitespaceValidation()

    def test_validate_trailing_whitespace(self):
        self.validate_and_compare(
            [
                'trailing space   ',
                'trailing tabs  ',
                '''trailing newline
                '''
            ],
            True,
            'is incorrectly failing on trailing whitespace'
        )

    def test_validate_leading_whitespace(self):
        self.validate_and_compare(
            [
                '   leading spaces',
                '   leading tabs',
                '''
                leading newline''',
            ],
            False,
            'does not detect leading whitespace'
        )

    def test_validate_middle_whitespace(self):
        self.validate_and_compare(
            [
                'middle spaces',
                'middle tabs',
                '''middle
                newline''',
            ],
            True,
            'is incorrectly failing on central whitespace'
        )


class TrailingWhitespace(ValidationTestBase):
    """
    Tests the TrailingWhitespaceValidation
    """

    def setUp(self):
        self.validator = TrailingWhitespaceValidation()
        super().setUp()

    def test_validate_trailing_whitespace(self):
        self.validate_and_compare(
            [
                'trailing space   ',
                'trailing tabs  ',
                '''trailing newline
                '''
            ],
            False,
            'is not detecting trailing whitespace'
        )

    def test_validate_leading_whitespace(self):
        self.validate_and_compare(
            [
                '   leading spaces',
                '   leading tabs',
                '''
                leading newline''',
            ],
            True,
            'is incorrectly failing on leading whitespace'
        )

    def test_validate_middle_whitespace(self):
        self.validate_and_compare(
            [
                'middle spaces',
                'middle tabs',
                '''middle
                newline''',
            ],
            True,
            'is incorrectly failing on central whitespace'
        )


class CanCallJson(ValidationTestBase):
    """
    Tests the CanCallValidation using json.loads
    """

    def setUp(self):
        self.validator = CanCallValidation(json.loads)

    def test_validate_valid_json(self):
        self.validate_and_compare(
            [
                '[1, 2, 3]',
                '{"a": 1.1, "b": 2.2, "c": 3.3}',
                '"string"'
            ],
            True,
            'is incorrectly failing on valid JSON'
        )

    def test_validate_invalid_json(self):
        self.validate_and_compare(
            [
                '[1, 2, 3',
                '{a: 1.1, b: 2.2, c: 3.3}',
                'string'
            ],
            False,
            'is not detecting invalid JSON'
        )


class CanCallLambda(ValidationTestBase):
    """
    Tests the CanCallValidation using a custom lambda function
    """

    def setUp(self):
        # Succeed if it's divisible by 2, otherwise cause an error
        self.validator = CanCallValidation(lambda x: False if x % 2 == 0 else 1 / 0)

    def test_validate_noerror(self):
        self.validate_and_compare(
            [
                2,
                4,
                6
            ],
            True,
            'is incorrectly failing on even numbers'
        )

    def test_validate_error(self):
        self.validate_and_compare(
            [
                1,
                3,
                5
            ],
            False,
            'should fail on odd numbers'
        )


class CanConvertInt(ValidationTestBase):
    """
    Tests CanConvertValidation using the int type
    """

    def setUp(self):
        self.validator = CanConvertValidation(int)

    def test_valid_int(self):
        self.validate_and_compare(
            [
                '1',
                '10',
                '999',
                '99999'
            ],
            True,
            'does not accept valid integers'
        )

    def test_invalid_int(self):
        self.validate_and_compare(
            [
                '1.0',
                '9.5',
                'abc',
                '1e-6'
            ],
            False,
            'accepts invalid integers'
        )


class InListCaseSensitive(ValidationTestBase):
    def setUp(self):
        self.validator = InListValidation(['a', 'b', 'c'])

    def test_valid_elements(self):
        self.validate_and_compare(
            [
                'a',
                'b',
                'c'
            ],
            True,
            'does not accept elements that are in the validation list'
        )

    def test_invalid_elements(self):
        self.validate_and_compare(
            [
                'aa',
                'bb',
                'd',
                'A',
                'B',
                'C'
            ],
            False,
            'accepts elements that are not in the validation list'
        )


class InListCaseInsensitive(ValidationTestBase):
    def setUp(self):
        self.validator = InListValidation(['a', 'b', 'c'], case_sensitive=False)

    def test_valid_elements(self):
        self.validate_and_compare(
            [
                'a',
                'b',
                'c',
                'A',
                'B',
                'C'
            ],
            True,
            'does not accept elements that are in the validation list'
        )

    def test_invalid_elements(self):
        self.validate_and_compare(
            [
                'aa',
                'bb',
                'd',
            ],
            False,
            'accepts elements that are not in the validation list'
        )


class DateFormat(ValidationTestBase):
    def setUp(self):
        self.validator = DateFormatValidation('%Y%m%d')

    def test_valid_dates(self):
        self.validate_and_compare(
            [
                '20160404',
                '00011212'
            ],
            True,
            'does not accept valid dates'
        )

    def test_invalid_dates(self):
        self.validate_and_compare(
            [
                '1/2/3456',
                'yyyymmdd',
                '11112233'
            ],
            False,
            'accepts invalid dates'
        )


class StringRegexMatch(ValidationTestBase):
    def setUp(self):
        self.validator = MatchesPatternValidation('^.+\.txt$')

    def test_valid_strings(self):
        self.validate_and_compare(
            [
                'pass.txt',
                'a.txt',
                'lots of words.txt'
            ],
            True,
            'does not accept strings matching the regex'
        )

    def test_invalid_strings(self):
        self.validate_and_compare(
            [
                'pass.TXT',
                '.txt',
                'lots of words.tx'
            ],
            False,
            'accepts strings that do not match the regex'
        )


class IsDistinct(ValidationTestBase):
    def setUp(self):
        self.validator = IsDistinctValidation()

    def test_valid_strings(self):
        self.validate_and_compare(
            [
                '1',
                '2',
                '3',
                '4'
            ],
            True,
            'does not accept unique strings'
        )

    def test_invalid_strings(self):
        validation = self.validator.validate(pd.Series([
            '1',
            '1',
            '3',
            '4'
        ]))

        self.assertTrue((validation == pd.Series([
            True,
            False,
            True,
            True
        ])).all(), 'did not identify the error')


class CompiledRegexMatch(ValidationTestBase):
    """
    Tests the MatchesRegexValidation, using a compiled regex
    """

    def setUp(self):
        self.validator = MatchesPatternValidation(re.compile('^.+\.txt$', re.IGNORECASE))

    def test_valid_strings(self):
        self.validate_and_compare(
            [
                'pass.txt',
                'a.TXT',
                'lots of words.tXt'
            ],
            True,
            'does not accept strings matching the regex'
        )

    def test_invalid_strings(self):
        self.validate_and_compare(
            [
                'pass.txtt',
                '.txt',
                'lots of words.tx'
            ],
            False,
            'accepts strings that do not match the regex'
        )


class InRange(ValidationTestBase):
    """
    Tests the InRangeValidation
    """

    def setUp(self):
        self.validator = InRangeValidation(7, 9)

    def test_valid_items(self):
        self.validate_and_compare(
            [
                7,
                8,
                7
            ],
            True,
            'does not accept integers in the correct range'
        )

    def test_invalid_items(self):
        self.validate_and_compare(
            [
                1,
                2,
                3
            ],
            False,
            'Incorrectly accepts integers outside of the range'
        )

    def test_valid_character_items(self):
        self.validate_and_compare(
            [
                7,
                "8",
                8
            ],
            True,
            "Does not accept integers provided as a string"
        )

    def test_invalid_character_items(self):
        self.validate_and_compare(
            [
                "seven",
                "eight",
            ],
            False,
            "Incorrectly accepts items with non numerical text"
        )


class Dtype(ValidationTestBase):
    """
    Tests the DtypeValidation
    """

    def setUp(self):
        self.validator = IsDtypeValidation(np.number)

    def test_valid_items(self):
        errors = self.validator.get_errors(pd.Series(
            [
                1,
                2,
                3
            ]))

        self.assertEqual(len(errors), 0)

    def test_invalid_items(self):
        errors = self.validator.get_errors(pd.Series(
            [
                'a',
                '',
                'c'
            ]))

        self.assertEqual(len(errors), 1)
        self.assertEqual(type(errors[0]), ValidationWarning)

    def test_schema(self):
        """
        Test this validation inside a schema, to ensure we get helpful error messages.
        In particular, we want to make sure that a ValidationWarning without a row number won't break the schema
        """
        df = pd.DataFrame(data={
            'wrong_dtype1': ['not_an_int'],
            'wrong_dtype2': [123],
            'wrong_dtype3': [12.5]
        })

        schema = Schema([
            Column('wrong_dtype1', [IsDtypeValidation(dtype('int64'))]),
            Column('wrong_dtype2', [IsDtypeValidation(dtype('float64'))]),
            Column('wrong_dtype3', [IsDtypeValidation(dtype('int64'))]),
        ])

        errors = schema.validate(df)

        self.assertEqual(
            sorted([str(x) for x in errors]),
            sorted([
                'The column wrong_dtype1 has a dtype of object which is not a subclass of the required type int64',
                'The column wrong_dtype2 has a dtype of int64 which is not a subclass of the required type float64',
                'The column wrong_dtype3 has a dtype of float64 which is not a subclass of the required type int64'
            ])
        )


class Negate(ValidationTestBase):
    """
    Tests the ~ operator on a MatchesPatternValidation
    """

    def setUp(self):
        self.validator = ~MatchesPatternValidation('fail')

    def test_valid_items(self):
        self.validate_and_compare(
            [
                'Pass',
                '1',
                'True'
            ],
            True,
            'Rejects values that should pass'
        )

    def test_invalid_items(self):
        self.validate_and_compare(
            [
                'fail',
                'thisfails',
                'failure'
            ],
            False,
            'Accepts values that should pass'
        )


class Or(ValidationTestBase):
    """
    Tests the | operator on two MatchesPatternValidations
    """

    def setUp(self):
        self.validator = MatchesPatternValidation('yes') | MatchesPatternValidation('pass')

    def test_valid_items(self):
        self.validate_and_compare(
            [
                'pass',
                'yes',
                'passyes',
                '345yes345'
            ],
            True,
            'Rejects values that should pass'
        )

    def test_invalid_items(self):
        self.validate_and_compare(
            [
                'fail',
                'YES',
                'YPESS'
            ],
            False,
            'Accepts values that should pass'
        )


class CustomMessage(ValidationTestBase):
    """
    Tests that custom error messages work as expected
    """

    def setUp(self):
        self.message = "UNUSUAL MESSAGE THAT WOULDN'T BE IN A NORMAL ERROR"

    def test_default_message(self):
        validator = InRangeValidation(min=4)
        for error in validator.get_errors(pd.Series(
                [
                    1,
                    2,
                    3
                ]
        ), Column('')):
            self.assertNotRegex(error.message, self.message, 'Validator not using the default warning message!')

    def test_custom_message(self):
        validator = InRangeValidation(min=4, message=self.message)
        for error in validator.get_errors(pd.Series(
                [
                    1,
                    2,
                    3
                ]
        ), Column('')):
            self.assertRegex(error.message, self.message, 'Validator not using the custom warning message!')


class GetErrorTests(ValidationTestBase):
    """
    Tests for float valued columns where allow_empty=True
    """

    def setUp(self):
        self.vals = [1.0, None, 3]

    def test_in_range_allow_empty_with_error(self):
        validator = InRangeValidation(min=4)
        errors = validator.get_errors(pd.Series(self.vals), Column('', allow_empty=True))
        self.assertEqual(len(errors), sum(v is not None for v in self.vals))

    def test_in_range_allow_empty_with_no_error(self):
        validator = InRangeValidation(min=0)
        errors = validator.get_errors(pd.Series(self.vals), Column('', allow_empty=True))
        self.assertEqual(len(errors), 0)

    def test_in_range_allow_empty_false_with_error(self):
        validator = InRangeValidation(min=4)
        errors = validator.get_errors(pd.Series(self.vals), Column('', allow_empty=False))
        self.assertEqual(len(errors), len(self.vals))


class PandasDtypeTests(ValidationTestBase):
    """
    Tests Series with various pandas dtypes that don't exist in numpy (specifically categories)
    """

    def setUp(self):
        self.validator = InListValidation(['a', 'b', 'c'], case_sensitive=False)

    def test_valid_elements(self):
        errors = self.validator.get_errors(pd.Series(['a', 'b', 'c', None, 'A', 'B', 'C'], dtype='category'),
                                           Column('', allow_empty=True))
        self.assertEqual(len(errors), 0)

    def test_invalid_empty_elements(self):
        errors = self.validator.get_errors(pd.Series(['aa', 'bb', 'd', None], dtype='category'),
                                           Column('', allow_empty=False))
        self.assertEqual(len(errors), 4)

    def test_invalid_and_empty_elements(self):
        errors = self.validator.get_errors(pd.Series(['a', None], dtype='category'),
                                           Column('', allow_empty=False))
        self.assertEqual(len(errors), 1)

    def test_invalid_elements(self):
        errors = self.validator.get_errors(pd.Series(['aa', 'bb', 'd'], dtype='category'),
                                           Column('', allow_empty=True))
        self.assertEqual(len(errors), 3)

class GetErrorAllowEmptyDatetimeTests(ValidationTestBase):
    """
    Tests for datetime valued columns where allow_empty=True
    """

    def setUp(self):
        match_val = datetime.datetime(2020, 11, 1)
        self.validator = CustomSeriesValidation(lambda s: s == match_val, 'did not match target date')

    def test_valid(self):
        series = pd.Series(['2020-11-01'], dtype='datetime64[ns]')
        errors = self.validator.get_errors(series, Column('', allow_empty=True))
        self.assertEqual(len(errors), 0)

    def test_valid_invalid(self):
        series = pd.Series(['2020-11-01', '2025-01-01'], dtype='datetime64[ns]')
        errors = self.validator.get_errors(series, Column('', allow_empty=True))
        self.assertEqual(len(errors), 1)

    def test_valid_invalid_empty(self):
        series = pd.Series(['2020-11-01', '2025-01-01', pd.NaT, np.NaN], dtype='datetime64[ns]')
        errors = self.validator.get_errors(series, Column('', allow_empty=True))
        self.assertEqual(len(errors), 1)
