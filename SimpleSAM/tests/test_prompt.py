import pytest
import numpy as np

from SimpleSAM.prompt import *


@pytest.fixture
def prompts_empty():
    return Prompts()

@pytest.fixture
def prompts_populated():
    return Prompts(
        point_coords = np.array([[4,6], [12, 35], [9, 10]]), 
        point_labels = np.array([1, 0, 1]),
        box = np.array([0, 0, 5, 5]))

def test_add_point_given_empty_adds_point(prompts_empty):
    new_point = [4, 6]
    prompts_empty.add_point(new_point, 1)
    assert new_point in prompts_empty.point_coords

def test_add_point_given_populated_adds_point(prompts_populated):
    new_point = [11, 12]
    prompts_populated.add_point(new_point, 1)
    assert new_point in prompts_populated.point_coords

def test_add_point_given_empty_adds_label(prompts_empty):
    new_point = [4, 6]
    prompts_empty.add_point(new_point, 1)
    assert prompts_empty.point_labels[-1] == 1

def test_add_point_given_populated_adds_label(prompts_populated):
    new_point = [2,3]
    prompts_populated.add_point(new_point, 1)
    assert prompts_populated.point_labels[-1] == 1

def test_remove_point_given_populated_removes_point_and_label(prompts_populated):
    test_index = 1
    test_point = prompts_populated.point_coords[test_index]
    test_label = prompts_populated.point_labels[test_index]
    num_coords_before = len(prompts_populated.point_coords)
    num_labels_before = len(prompts_populated.point_labels)

    prompts_populated.remove_point(test_point)
    assert len(prompts_populated.point_coords) < num_coords_before 
    assert len(prompts_populated.point_labels) < num_labels_before 
    assert test_point not in prompts_populated.point_coords

def test_remove_point_given_populate_when_point_not_present_does_nothing(prompts_populated):
    test_point = np.array([99, 99])
    num_coords_before = len(prompts_populated.point_coords)
    num_labels_before = len(prompts_populated.point_labels)

    prompts_populated.remove_point(test_point)
    assert len(prompts_populated.point_coords) == num_coords_before 
    assert len(prompts_populated.point_labels) == num_labels_before 

