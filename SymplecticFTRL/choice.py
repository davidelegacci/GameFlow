import numpy as np
import sympy as sp
from gamelab.finitegames import NFG_bare
from aspera import spacemaker as sm

# restructure using aspera.spacemaker

# knows nothing about game payoff

class Logit():
	"""Knows nothing about the game"""
	def __init__(self, dimension, primal_letter, dual_letter, player_label):
 
		self.dual_space = sm.Space(dual_letter.capitalize(), dimension, f'{dual_letter}_', 0 )
		self.primal_space = sm.Space(primal_letter.capitalize(), dimension, f'{primal_letter}_', 0 )
		exp_y = np.array([ sp.exp(y_a) for y_a in self.dual_space.vars ])
		self.Q = sm.Map( f'Q_{player_label}', self.dual_space, self.primal_space, exp_y / sum(exp_y))


class QuotientLogit(Logit):
	"""Knows nothing about the game"""
	def __init__(self, dimension, primal_letter, dual_letter, player_label, quot_letter):
		super().__init__(dimension, primal_letter, dual_letter, player_label)
		self.quot_letter = quot_letter
		self.dummy_dual = 0

		self.quot_space = sm.Space(quot_letter.capitalize(), dimension, f'{quot_letter}_', 0 )

		self.eff_primal_space = sm.Space(primal_letter.capitalize(), dimension-1, f'{primal_letter}_', 1 )
		self.eff_quot_space = sm.Space(quot_letter.capitalize(), dimension-1, f'{quot_letter}_', 1 )

		Q_hat_numerator = np.array([1] + [ sp.exp(z_a) for z_a in self.quot_space.vars[1:] ])
		self.Q = sm.Map( f'q_{player_label}', self.quot_space, self.primal_space, Q_hat_numerator / sum(Q_hat_numerator))

		#eff_Q_hat_numerator = np.array( [ sp.exp(z_a) for z_a in self.quot_space.vars[1:] ])
		self.eff_Q = sm.Map( f'q_{player_label}', self.eff_quot_space, self.eff_primal_space, self.Q.ex[1:] )


class MultiLogit():
	"""docstring for Choice"""
	def __init__(self, primal_letter, dual_letter, skeleton):
		self.skeleton = skeleton
		self.bare = NFG_bare(skeleton, "numeric", primal_letter)
		self.primal_vars_play = self.bare.strat_play
		self.dual_vars_play = self.bare.make_vars_play(dual_letter, skeleton)

		logits = [ ]
		for i, Ai in enumerate(skeleton):
			logits.append( Logit( Ai, f'{self.primal_vars_play[i]}', f'{self.dual_vars_play[i]}', i ) )

		# spacemaker.Space instances
		self.primal_space = sm.Product.multispaces( [ logit.primal_space  for logit in logits] )
		self.dual_space = sm.Product.multispaces( [ logit.dual_space  for logit in logits] )
		
		# spacemaker.Map instance
		self.Q = sm.Product.multimaps( [ logit.Q  for logit in logits ])

class QuotientMultiLogit(MultiLogit):
	"""docstring for Choice"""
	def __init__(self, primal_letter, dual_letter, skeleton, quot_letter):
		super().__init__(primal_letter, dual_letter, skeleton)

		self.quot_vars_play = self.bare.make_vars_play(quot_letter, skeleton)

		quot_logits = [ ]
		for i, Ai in enumerate(skeleton):
			quot_logits.append( QuotientLogit( Ai, f'{self.primal_vars_play[i]}', f'{self.dual_vars_play[i]}', i, f'{self.quot_vars_play[i]}' ) )

		# spacemaker.Space instances
		self.quot_space = sm.Product.multispaces( [ qlogit.quot_space  for qlogit in quot_logits] )
		self.eff_quot_space = sm.Product.multispaces( [ qlogit.eff_quot_space  for qlogit in quot_logits] )
		
		# spacemaker.Map instance
		self.Q = sm.Product.multimaps( [ qlogit.Q  for qlogit in quot_logits ])
		self.eff_Q = sm.Product.multimaps( [ qlogit.eff_Q  for qlogit in quot_logits ])

# ML = MultiLogit('y', [2,2] )

# print(ML.Q.ex)





