from django.test import TestCase


import model1.models as models
import numpy as np


class ModelTest(TestCase):
    # test if model returns a number
    def test_normalPrediction(self):
        # create prediction object
        model = models.PredictionHistGradientBoostingRegression(
            living_space=100,
            floor_space=100,
            type="detached-house",
            plot_area=100,
            rooms=3,
            zip_code=1000,
            last_refurbishment=2000,
            year_built=2000,
        )
        # get prediction
        prediction = model.predict()
        # check if prediction is a number
        self.assertEqual(type(prediction), int)

    # test if model returns a number with minimal input
    def test_minimalPrediction(self):
        # create prediction object
        model = models.PredictionHistGradientBoostingRegression(
            living_space=100,
            type="detached-house",
            rooms=3,
            zip_code=1000,
            floor_space=np.nan,
            plot_area=np.nan,
            last_refurbishment=np.nan,
            year_built=np.nan,
        )
        # get prediction
        prediction = model.predict()
        # check if prediction is a number
        self.assertEqual(type(prediction), int)

    # test if model returns None missing the value for living_space
    def test_missingLivingSpace(self):
        # create prediction object
        model = models.PredictionHistGradientBoostingRegression(
            living_space=np.nan,
            type="detached-house",
            rooms=3,
            zip_code=1000,
            floor_space=np.nan,
            plot_area=np.nan,
            last_refurbishment=np.nan,
            year_built=np.nan,
        )
        # get prediction
        prediction = model.predict()
        # should give None
        self.assertEqual(prediction, None)

    # test if model returns None missing the value for type
    def test_missingType(self):
        # create prediction object
        model = models.PredictionHistGradientBoostingRegression(
            living_space=100,
            type=None,
            rooms=3,
            zip_code=1000,
            floor_space=np.nan,
            plot_area=np.nan,
            last_refurbishment=np.nan,
            year_built=np.nan,
        )
        # get prediction
        prediction = model.predict()
        # should give None
        self.assertEqual(prediction, None)

    # test if model returns a None missing the value for rooms
    def test_missingRooms(self):
        # create prediction object
        model = models.PredictionHistGradientBoostingRegression(
            living_space=100,
            type="detached-house",
            rooms=np.nan,
            zip_code=1000,
            floor_space=np.nan,
            plot_area=np.nan,
            last_refurbishment=np.nan,
            year_built=np.nan,
        )
        # get prediction
        prediction = model.predict()
        # should give None
        self.assertEqual(prediction, None)

    # test if model throws an error having a zip code of < 1000
    def test_zipCodeTooSmall(self):
        # create prediction object
        model = models.PredictionHistGradientBoostingRegression(
            living_space=100,
            type="detached-house",
            rooms=3,
            zip_code=999,
            floor_space=np.nan,
            plot_area=np.nan,
            last_refurbishment=np.nan,
            year_built=np.nan,
        )
        try:
            # get prediction
            prediction = model.predict()
        except KeyError:
            # should give KeyError
            self.assertEqual(True, True)
        else:
            # should not give KeyError
            self.assertEqual(True, False)

    # test if model throws an error having a zip code of > 9999
    def test_zipCodeTooBig(self):
        # create prediction object
        model = models.PredictionHistGradientBoostingRegression(
            living_space=100,
            type="detached-house",
            rooms=3,
            zip_code=10000,
            floor_space=np.nan,
            plot_area=np.nan,
            last_refurbishment=np.nan,
            year_built=np.nan,
        )
        try:
            # get prediction
            prediction = model.predict()
        except KeyError:
            # should give KeyError
            self.assertEqual(True, True)
        else:
            # should not give KeyError
            self.assertEqual(True, False)

    # test if model returns a number having a zip code of 1000
    def test_zipCode1000(self):
        # create prediction object
        model = models.PredictionHistGradientBoostingRegression(
            living_space=100,
            type="detached-house",
            rooms=3,
            zip_code=1000,
            floor_space=np.nan,
            plot_area=np.nan,
            last_refurbishment=np.nan,
            year_built=np.nan,
        )
        # get prediction
        prediction = model.predict()
        # should give a number
        self.assertEqual(type(prediction), int)

    # test if model return None using an invalid type
    def test_invalidType(self):
        # create prediction object
        model = models.PredictionHistGradientBoostingRegression(
            living_space=100,
            type="cashmoneyhouse",
            rooms=3,
            zip_code=1000,
            floor_space=np.nan,
            plot_area=np.nan,
            last_refurbishment=np.nan,
            year_built=np.nan,
        )
        # get prediction
        prediction = model.predict()
        # should give None
        self.assertEqual(prediction, None)

    # test if model return same number for two different zip codes
    def test_validZipCode(self):
        # create prediction object
        model = models.PredictionHistGradientBoostingRegression(
            living_space=100,
            type="detached-house",
            rooms=3,
            zip_code=1000,
            floor_space=np.nan,
            plot_area=np.nan,
            last_refurbishment=np.nan,
            year_built=np.nan,
        )
        model2 = models.PredictionHistGradientBoostingRegression(
            living_space=100,
            type="detached-house",
            rooms=3,
            zip_code=8000,
            floor_space=np.nan,
            plot_area=np.nan,
            last_refurbishment=np.nan,
            year_built=np.nan,
        )
        # get prediction
        prediction = model.predict()
        prediction2 = model2.predict()
        # should give different number
        self.assertNotEqual(prediction, prediction2)

    # test if model return same number for two different rooms
    def test_validRooms(self):
        # create prediction object
        model = models.PredictionHistGradientBoostingRegression(
            living_space=100,
            type="detached-house",
            rooms=3,
            zip_code=1000,
            floor_space=np.nan,
            plot_area=np.nan,
            last_refurbishment=np.nan,
            year_built=np.nan,
        )
        model2 = models.PredictionHistGradientBoostingRegression(
            living_space=100,
            type="detached-house",
            rooms=6,
            zip_code=1000,
            floor_space=np.nan,
            plot_area=np.nan,
            last_refurbishment=np.nan,
            year_built=np.nan,
        )
        # get prediction
        prediction = model.predict()
        prediction2 = model2.predict()
        # should give different number
        self.assertNotEqual(prediction, prediction2)

    # test if model return same number for two different types
    def test_validType(self):
        # create prediction object
        model = models.PredictionHistGradientBoostingRegression(
            living_space=100,
            type="detached-house",
            rooms=3,
            zip_code=1000,
            floor_space=np.nan,
            plot_area=np.nan,
            last_refurbishment=np.nan,
            year_built=np.nan,
        )
        model2 = models.PredictionHistGradientBoostingRegression(
            living_space=100,
            type="villa",
            rooms=3,
            zip_code=1000,
            floor_space=np.nan,
            plot_area=np.nan,
            last_refurbishment=np.nan,
            year_built=np.nan,
        )
        # get prediction
        prediction = model.predict()
        prediction2 = model2.predict()
        # should give different number
        self.assertNotEqual(prediction, prediction2)
