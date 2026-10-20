"""Sensory-only approach, search and stopping behavior."""

from flybrain.training.food_teacher import FoodTeacher
from flybrain.world.food import FoodObservation
from flybrain.world.fly import MotorCommand


def test_teacher_slows_with_mean_odor_and_stops_on_contact():
    teacher = FoodTeacher()
    far = teacher.command(FoodObservation(.02, .02, False))
    middle = teacher.command(FoodObservation(.5, .5, False))
    near = teacher.command(FoodObservation(1.5, 1.5, False))
    assert .6 < far.forward < .7
    assert .2 < near.forward < middle.forward < far.forward
    assert teacher.command(FoodObservation(2, 1.5, True)) == MotorCommand()
    assert teacher.command(FoodObservation(2, 1.5, True)) == MotorCommand()


def test_teacher_turns_toward_stronger_antenna_and_searches_on_falling_equal_odor():
    teacher = FoodTeacher()
    assert teacher.command(FoodObservation(.03, .05, False)).turn > 0
    teacher.reset()
    teacher.command(FoodObservation(.025, .025, False))
    search = teacher.command(FoodObservation(.024, .024, False))
    assert search.turn > 0
    teacher.reset()
    assert teacher.command(FoodObservation(.025, .025, False)).turn == 0
