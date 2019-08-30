import numpy as np
import pandas as pd
from itertools import combinations

def set_sniffer(attr1, attr2, attr3):
	if attr1 == attr2 and attr2 == attr3:
		return True
	elif (attr1 != attr2) and (attr2 != attr3) and (attr3 != attr1):
		return True
	else:
		return False

def check_cards(card1, card2, card3):
	color_check = set_sniffer(card1['colors'], card2['colors'], card3['colors'])
	num_check = set_sniffer(card1['nums'], card2['nums'], card3['nums'])
	shape_check = set_sniffer(card1['shapes'], card2['shapes'], card3['shapes'])
	shade_check = set_sniffer(card1['shades'], card2['shades'], card3['shades'])
	return color_check and num_check and shape_check and shade_check

def check_set(cards):
	indices, card_stack = [],[]
	for index, card in cards.iterrows():
		indices.append(index)
		card_stack.append(card)
	card_combos = list(combinations(indices,3)) # map the combinations to the indices
	for combination in card_combos:
		if check_cards(card_stack[combination[0]],card_stack[combination[1]],card_stack[combination[2]]):
			return combination
		else:
			pass