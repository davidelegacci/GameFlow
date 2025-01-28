import numpy as np
import sympy as sp

# restructure using aspera.spacemaker

class SFTRL():	
	"""docstring for SFTRL"""
	def __init__(self, map_payfield, map_choice, primal_vars, dual_vars, name):

		self.map_payfield = map_payfield 		# aspera.spacemaker.Map instance
		self.map_choice = map_choice 			# aspera.spacemaker.Map instance
		#self.primal_vars = primal_vars 	# flat symbolic primal variables, array of size 4; not really used
		self.dual_vars = dual_vars		# flat symbolic dual variables, array of size 4
		self.name = name

		# Make V composed Q symbolic
		#self.map_VQ = [  sp.simplify(v.subs(dict(zip(self.primal_vars, self.map_choice)))) for v in self.map_payfield ]
		self.map_VQ = self.map_payfield.compose(self.map_choice) # aspera.spacemaker.Map instance

		# Make d(VQ)
		self.sym_dVQ = self.diff_oneform(self.map_VQ.ex, self.map_VQ.vars)


		# # Make d(VQ)(VQ)
		self.sym_dVQ_VQ = sp.simplify(self.sym_dVQ * self.map_VQ.ex)

		# #----------
		# # Geometric version --> must do in Z; now just pull, no sharp
		# #----------
		self.sym_JQ = self.map_choice.ex.jacobian(self.map_choice.vars)
		assert self.sym_JQ == self.sym_JQ.T
		self.sym_QpullV = sp.simplify( self.sym_JQ * self.map_VQ.ex) # Would be Transpose, but symmetric
		self.sym_d_QpullV = sp.simplify(self.diff_oneform(self.sym_QpullV, self.dual_vars))
		self.sym_symplectic_1form = sp.simplify(self.sym_d_QpullV * self.map_VQ.ex)
		# -----------------------------------------------------------------------------


		# Lambdify stuff


		# Choice
		#self.choice = sp.lambdify(self.dual_vars, self.map_choice.ex)
		self.choice = self.map_choice.lambdify_map()

		# Payfield
		#self.payfield = sp.lambdify(self.primal_vars,  self.map_payfield.ex )
		self.payfield = self.map_payfield.lambdify_map()

		# Standard FTRL
		self.VQ = self.map_VQ.lambdify_map()

		# Symplectic correction d(VQ)(VQ); it is not a Map but just a Sympy matrix, so need list() to flatten output
		self.dVQ_VQ = sp.lambdify(self.dual_vars, list(self.sym_dVQ_VQ))
		self.symplectic_1form = sp.lambdify(self.dual_vars, list(self.sym_symplectic_1form))
		self.QpullV = sp.lambdify(self.dual_vars, list(self.sym_QpullV))


	def diff_oneform(self, oneform, variables):
		# oneform is sympy flat Matrix
		J = oneform.jacobian(variables)
		return J.T - J


	# Symplectic FTRL
	def sftrl_dyn(self, y, t, lam):
		return np.array( self.VQ(*y) ) + lam * np.array( self.dVQ_VQ(*y) )


	# Pulled symplectic FTRL
	def sftrl_dyn_2(self, y, t, lam):
		return np.array( self.QpullV(*y) ) + lam * np.array( self.symplectic_1form(*y) )

# ----------------------------------------------------------------------------------------


class ZSFTRL():	
	"""docstring for SFTRL"""
	def __init__(self, map_payfield, map_choice, primal_vars, quot_vars, name):

		self.map_payfield = map_payfield 		# aspera.spacemaker.Map instance; this is reduced payfield, already descended
		self.map_choice = map_choice 			# aspera.spacemaker.Map instance; this is Q hat, already descended
		#self.primal_vars = primal_vars 	# flat symbolic primal variables; not really used
		self.quot_vars = quot_vars		# flat symbolic dual variables
		self.name = name

		# Make V composed Q symbolic; this is Z field
		self.map_VQ = self.map_payfield.compose(self.map_choice) # aspera.spacemaker.Map instance

		# Make d(VQ)
		self.sym_dVQ = self.diff_oneform(self.map_VQ.ex, self.map_VQ.vars)


		# # Make d(VQ)(VQ)
		self.sym_dVQ_VQ = sp.simplify(self.sym_dVQ * self.map_VQ.ex)

		# #----------
		# # Geometric version --> must do in Z in local coordinates
		# #----------
		self.geometric_version = False
		self.sym_JQ = self.map_choice.ex.jacobian(self.map_choice.vars)
		if self.sym_JQ == self.sym_JQ.T:
			self.geometric_version = True
			print('Init geometric version')
			self.sym_QpullV = sp.simplify( self.sym_JQ * self.map_VQ.ex) # Would be Transpose, but symmetric
			print('pull done...')
			self.sym_d_QpullV = sp.simplify(self.diff_oneform(self.sym_QpullV, self.quot_vars))
			print('diff done...')
			self.sym_symplectic_1form = sp.simplify(self.sym_d_QpullV * self.map_VQ.ex)
			print('product done...')
			self.sym_symplectic_vfield = sp.simplify(self.sym_JQ.inv() * self.sym_symplectic_1form)
			print('Inverse done, Symbolic computations over')

		# -----------------------------------------------------------------------------


		# Lambdify stuff


		# Choice
		#self.choice = sp.lambdify(self.quot_vars, self.map_choice.ex)
		self.choice = self.map_choice.lambdify_map()

		# Payfield
		#self.payfield = sp.lambdify(self.primal_vars,  self.map_payfield.ex )
		# self.payfield = self.map_payfield.lambdify_map()

		# Standard FTRL
		self.VQ = self.map_VQ.lambdify_map()

		# Symplectic correction d(VQ)(VQ); it is not a Map but just a Sympy matrix, so need list() to flatten output
		self.dVQ_VQ = sp.lambdify(self.quot_vars, list(self.sym_dVQ_VQ))

		
		if self.geometric_version:
			# Geometric Symplectic correction d(VQ)(VQ); it is not a Map but just a Sympy matrix, so need list() to flatten output
			self.symplectic_vfield = sp.lambdify(self.quot_vars, list(self.sym_symplectic_vfield))
			self.QpullV = sp.lambdify(self.quot_vars, list(self.sym_QpullV))


	def diff_oneform(self, oneform, variables):
		# oneform is sympy flat Matrix
		J = oneform.jacobian(variables)
		return J.T - J


	# Symplectic FTRL
	def sftrl_dyn(self, z, t, lam):
		# works in full coords (eg 4 in 2x2)
		return np.array( self.VQ(*z) )  + lam * np.array( self.dVQ_VQ(*z) )


	# Geometric Symplectic FTRL; if geometric version
	def geom_sftrl_dyn(self, z, t, lam):
		# works in local coords (eg 2 in 2x2)
		return np.array( self.VQ(*z) )  + lam * np.array( self.symplectic_vfield(*z) )

	# Geometric Symplectic FTRL; if geometric version; pull also first term; now this is 1-form, sharp would cancel pulling by inverse!
	def geom_sftrl_dyn_2(self, z, t, lam):
		# works in local coords (eg 2 in 2x2)
		return np.array( self.QpullV(*z) )  + lam * np.array( self.symplectic_vfield(*z) )

# ----------------------------------------------------------------------------------------
		