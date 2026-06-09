'''
	******************************************************************************************
	    Assembly:                Boo
	    Filename:                Boo.py
	    Author:                  Terry D. Eppler
	    Created:                 05-31-2022
	
	    Last Modified By:        Terry D. Eppler
	    Last Modified On:        05-01-2025
	******************************************************************************************
	<copyright file="gpt.md" company="Terry D. Eppler">
	
	           Boo is a df analysis tool integrating various Generative GPT, GptText-Processing, and
	           Machine-Learning algorithms for federal analysts.
	           Copyright ©  2022  Terry Eppler
	
	   Permission is hereby granted, free of charge, to any person obtaining a copy
	   of this software and associated documentation files (the “Software”),
	   to deal in the Software without restriction,
	   including without limitation the rights to use,
	   copy, modify, merge, publish, distribute, sublicense,
	   and/or sell copies of the Software,
	   and to permit persons to whom the Software is furnished to do so,
	   subject to the following conditions:
	
	   The above copyright notice and this permission notice shall be included in all
	   copies or substantial portions of the Software.
	
	   THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
	   INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
	   FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
	   IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
	   DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
	   ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
	   DEALINGS IN THE SOFTWARE.
	
	   You can contact me at:  terryeppler@gmail.com or eppler.terry@epa.gov
	
	</copyright>
	<summary>
	  Boo.py
	</summary>
	******************************************************************************************
'''
from __future__ import annotations
import json
import os
from pathlib import Path
import tiktoken
from openai import OpenAI
from typing import Optional, List, Dict, Any
from openai.types.responses import Response
import base64
from openai.types import CreateEmbeddingResponse, VectorStore, FileObject
from boogr import Error, Logger
import config as cfg
import tempfile

def throw_if( name: str, value: object ) -> None:
	"""Throw if.
	
	Purpose:
		Validates a required value before a provider or application operation proceeds. The
		function raises a ValueError when the supplied value is missing, blank, or empty.
	
	Args:
		name (str): Name value used by the operation.
		value (object): Value value used by the operation.
	
	Raises:
		ValueError: Raised when required input is missing or invalid.
	"""
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be empty!' )
	
	if isinstance( value, str ) and not value.strip( ):
		raise ValueError( f'Argument "{name}" cannot be empty!' )
	
	if isinstance( value, (list, tuple, dict, set) ) and len( value ) == 0:
		raise ValueError( f'Argument "{name}" cannot be empty!' )

def encode_image( image_path: str ) -> str:
	"""Encode image.
	
	Purpose:
		Reads a local image file and converts its bytes into a base64-encoded string. The
		encoded value is used by image and vision workflows that require inline image content.
	
	Args:
		image_path (str): Image path value used by the operation.
	
	Returns:
		Base64-encoded image content.
	"""
	with open( image_path, "rb" ) as image_file:
		return base64.b64encode( image_file.read( ) ).decode( 'utf-8' )

class GPT:
	"""Provide GPT workflow support.
	
	Purpose:
		Provides the shared OpenAI wrapper base used by Gipity provider workflows. The class
		stores common model, prompt, request, response, and compatibility fields inherited by
		text, image, audio, embedding, file, and vector-store wrappers.
	
	Attributes:
		api_key (Optional[str]): Api key retained by the provider wrapper.
		client (Optional[OpenAI]): Client retained by the provider wrapper.
		prompt (Optional[str]): Prompt retained by the provider wrapper.
		temperature (Optional[float]): Temperature retained by the provider wrapper.
		top_percent (Optional[float]): Top percent retained by the provider wrapper.
		frequency_penalty (Optional[float]): Frequency penalty retained by the provider wrapper.
		presence_penalty (Optional[float]): Presence penalty retained by the provider wrapper.
		max_tokens (Optional[int]): Max tokens retained by the provider wrapper.
		stops (Optional[List[str]]): Stops retained by the provider wrapper.
		store (Optional[bool]): Store retained by the provider wrapper.
		stream (Optional[bool]): Stream retained by the provider wrapper.
		background (Optional[bool]): Background retained by the provider wrapper.
		number (Optional[int]): Number retained by the provider wrapper.
		response_format (Optional[Dict[str, str]]): Response format retained by the provider wrapper.
		context (Optional[List[Dict[str, str]]]): Context retained by the provider wrapper.
		instructions (Optional[str]): Instructions retained by the provider wrapper.
	"""
	api_key: Optional[ str ]
	client: Optional[ OpenAI ]
	prompt: Optional[ str ]
	temperature: Optional[ float ]
	top_percent: Optional[ float ]
	frequency_penalty: Optional[ float ]
	presence_penalty: Optional[ float ]
	max_tokens: Optional[ int ]
	stops: Optional[ List[ str ] ]
	store: Optional[ bool ]
	stream: Optional[ bool ]
	background: Optional[ bool ]
	number: Optional[ int ]
	response_format: Optional[ Dict[ str, str ] ]
	context: Optional[ List[ Dict[ str, str ] ] ]
	instructions: Optional[ str ]
	
	def __init__( self ):
		"""Initialize instance.
		
		Purpose:
			Initializes the GPT object with default configuration, runtime state, provider settings,
			and compatibility fields. This constructor prepares the instance for later method calls
			without performing external work beyond local attribute assignment.
		"""
		self.api_key = cfg.OPENAI_API_KEY
		self.model = None
		self.client = None
		self.number = None
		self.stops = [ ]
		self.response_format = { }
		self.number = None
		self.temperature = None
		self.top_percent = None
		self.frequency_penalty = None
		self.presence_penalty = None
		self.max_tokens = None
		self.prompt = None
		self.store = None
		self.stream = None
		self.background = None
		self.instructions = None
		self.context = [ ]

class Chat( GPT ):
	"""Provide Chat workflow support.
	
	Purpose:
		Provides a stable wrapper around OpenAI Responses API text-generation workflows. The
		class manages model options, request construction, tool configuration, reasoning
		settings, response metadata, and conversation compatibility fields used by Gipity Text
		mode.
	
	Attributes:
		include (Optional[List[str]]): Include retained by the provider wrapper.
		tool_choice (Optional[str]): Tool choice retained by the provider wrapper.
		previous_id (Optional[str]): Previous id retained by the provider wrapper.
		conversation_id (Optional[str]): Conversation id retained by the provider wrapper.
		parallel_tools (Optional[bool]): Parallel tools retained by the provider wrapper.
		max_tools (Optional[int]): Max tools retained by the provider wrapper.
		input (Optional[List[Dict[str, Any]] | str]): Input retained by the provider wrapper.
		tools (Optional[List[Dict[str, Any]]]): Tools retained by the provider wrapper.
		reasoning (Optional[Dict[str, str]]): Reasoning retained by the provider wrapper.
		image_url (Optional[str]): Image url retained by the provider wrapper.
		image_path (Optional[str]): Image path retained by the provider wrapper.
		file_url (Optional[str]): File url retained by the provider wrapper.
		file_path (Optional[str]): File path retained by the provider wrapper.
		allowed_domains (Optional[List[str]]): Allowed domains retained by the provider wrapper.
		max_search_results (Optional[int]): Max search results retained by the provider wrapper.
		output_text (Optional[str]): Output text retained by the provider wrapper.
		vector_stores (Optional[Dict[str, str]]): Vector stores retained by the provider wrapper.
		files (Optional[Dict[str, str]]): Files retained by the provider wrapper.
		content (Optional[str]): Content retained by the provider wrapper.
		vector_store_ids (Optional[List[str]]): Vector store ids retained by the provider wrapper.
		file_ids (Optional[List[str]]): File ids retained by the provider wrapper.
		response (Optional[Response]): Response retained by the provider wrapper.
		file (Optional[FileObject]): File retained by the provider wrapper.
		purpose (Optional[str]): Purpose retained by the provider wrapper.
	"""
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	previous_id: Optional[ str ]
	conversation_id: Optional[ str ]
	parallel_tools: Optional[ bool ]
	max_tools: Optional[ int ]
	input: Optional[ List[ Dict[ str, Any ] ] | str ]
	tools: Optional[ List[ Dict[ str, Any ] ] ]
	reasoning: Optional[ Dict[ str, str ] ]
	image_url: Optional[ str ]
	image_path: Optional[ str ]
	file_url: Optional[ str ]
	file_path: Optional[ str ]
	allowed_domains: Optional[ List[ str ] ]
	max_search_results: Optional[ int ]
	output_text: Optional[ str ]
	vector_stores: Optional[ Dict[ str, str ] ]
	files: Optional[ Dict[ str, str ] ]
	content: Optional[ str ]
	vector_store_ids: Optional[ List[ str ] ]
	file_ids: Optional[ List[ str ] ]
	response: Optional[ Response ]
	file: Optional[ FileObject ]
	purpose: Optional[ str ]
	
	def __init__( self, model: str = 'gpt-5-nano', prompt: str = None, temperature: float = None,
			top_p: float = None, presense: float = None, presence: float = None, store: bool = None,
			stream: bool = None, stops: List[ str ] = None,
			response_format: Dict[ str, Any ] = None,
			number: int = None, instruct: str = None, context: List[ Dict[ str, str ] ] = None,
			allowed_domains: List[ str ] = None, include: List[ str ] = None,
			tools: List[ Dict[ str, Any ] ] = None, max_tools: int = None,
			tool_choice: str = None, file_path: str = None, background: bool = None,
			is_parallel: bool = None, max_tokens: int = None, frequency: float = None,
			input: List[ Dict[ str, Any ] ] = None, file_ids: List[ str ] = None,
			previous_id: str = None, conversation_id: str = None,
			reasoning: Dict[ str, str ] | str = None, output_text: str = None,
			max_search_results: int = None, content: str = None,
			vector_store_ids: List[ str ] = None ):
		"""Initialize instance.
		
		Purpose:
			Initializes the Chat object with default configuration, runtime state, provider
			settings, and compatibility fields. This constructor prepares the instance for later
			method calls without performing external work beyond local attribute assignment.
		
		Args:
			model (str): Model value used by the operation.
			prompt (str): Prompt value used by the operation.
			temperature (float): Temperature value used by the operation.
			top_p (float): Top p value used by the operation.
			presense (float): Presense value used by the operation.
			presence (float): Presence value used by the operation.
			store (bool): Store value used by the operation.
			stream (bool): Stream value used by the operation.
			stops (List[str]): Stops value used by the operation.
			response_format (Dict[str, Any]): Response format value used by the operation.
			number (int): Number value used by the operation.
			instruct (str): Instruct value used by the operation.
			context (List[Dict[str, str]]): Context value used by the operation.
			allowed_domains (List[str]): Allowed domains value used by the operation.
			include (List[str]): Include value used by the operation.
			tools (List[Dict[str, Any]]): Tools value used by the operation.
			max_tools (int): Max tools value used by the operation.
			tool_choice (str): Tool choice value used by the operation.
			file_path (str): File path value used by the operation.
			background (bool): Background value used by the operation.
			is_parallel (bool): Is parallel value used by the operation.
			max_tokens (int): Max tokens value used by the operation.
			frequency (float): Frequency value used by the operation.
			input (List[Dict[str, Any]]): Input value used by the operation.
			file_ids (List[str]): File ids value used by the operation.
			previous_id (str): Previous id value used by the operation.
			conversation_id (str): Conversation id value used by the operation.
			reasoning (Dict[str, str] | str): Reasoning value used by the operation.
			output_text (str): Output text value used by the operation.
			max_search_results (int): Max search results value used by the operation.
			content (str): Content value used by the operation.
			vector_store_ids (List[str]): Vector store ids value used by the operation.
		"""
		super( ).__init__( )
		self.api_key = cfg.OPENAI_API_KEY
		self.client = None
		self.model = model
		self.prompt = prompt
		self.number = number
		self.response_format = response_format if response_format is not None else { }
		self.temperature = temperature
		self.top_percent = top_p
		self.allowed_domains = allowed_domains if allowed_domains is not None else [ ]
		self.frequency_penalty = frequency
		self.presence_penalty = presence if presence is not None else presense
		self.max_tokens = max_tokens
		self.context = context if context is not None else [ ]
		self.stream = stream
		self.store = store
		self.instructions = instruct
		self.stops = stops if stops is not None else [ ]
		self.background = background
		self.input = input if input is not None else [ ]
		self.include = include if include is not None else [ ]
		self.output_text = output_text
		self.max_tools = max_tools
		self.vector_store_ids = vector_store_ids if vector_store_ids is not None else [ ]
		self.file_ids = file_ids if file_ids is not None else [ ]
		self.tools = tools if tools is not None else [ ]
		self.previous_id = previous_id
		self.conversation_id = conversation_id
		self.reasoning = reasoning
		self.parallel_tools = is_parallel
		self.tool_choice = tool_choice
		self.response = None
		self.file = None
		self.file_url = file_path
		self.file_path = file_path
		self.image_url = None
		self.content = content
		self.max_search_results = max_search_results
		self.purpose = None
		self.request = { }
		self.messages = [ ]
		self.built_tools = [ ]
		self.stream_requested = False
		self.background_requested = False
		self.effective_context = [ ]
		self.vector_stores = {
				'Governance': 'vs_6a1850a9bdc08191912353eedf59aede',
				'Public Laws': 'vs_699506f7d5348191990e0557c717fa9d',
				'Explanatory Statements': 'vs_699505df9ac48191a525c0ecb86fef66',
				'Army Techniques Publications': 'vs_699356ef052c81918da14c4ed3bcea17',
				'Army Field Manuals': 'vs_69935542863481918d150c1e89c38633',
				'Army Regulations': 'vs_6993550488408191919cd70968ba8be8',
				'DoD Armory': 'vs_697f86ad98888191b967685ae558bfc0',
				'Army Style Guides': 'vs_68f4efd7d4c4819191458dd6cde6f2cc',
				'Apportionments': 'vs_68a34aaff93481918c3b3fef8c4e8fea',
				'Financial Regulations': 'vs_712r5W5833G6aLxIYIbuvVcK' }
		
		self.files = {
				'Account_Balances.csv': 'file-U6wFeRGSeg38Db5uJzo5sj',
				'SF133.csv': 'file-WT2h2F5SNxqK2CxyAMSDg6',
				'Authority.csv': 'file-Qi2rw2QsdxKBX1iiaQxY3m',
				'Outlays.csv': 'file-GHEwSWR7ezMvHrQ3X648wn',
		}
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Get model options.
		
		Purpose:
			Returns the model options exposed by the Chat wrapper. The property centralizes UI
			option values and keeps application selectors aligned with the provider-specific
			implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return [ 'gpt-5.4', 'gpt-5.4-mini', 'gpt-5.4-nano', 'gpt-5.1', 'gpt-5',
		         'gpt-5-mini', 'gpt-5-nano', 'gpt-4.1', 'gpt-4.1-mini',
		         'gpt-4.1-nano', 'gpt-4o', 'gpt-4o-mini', ]
	
	@property
	def include_options( self ) -> List[ str ] | None:
		"""Get include options.
		
		Purpose:
			Returns the include options exposed by the Chat wrapper. The property centralizes UI
			option values and keeps application selectors aligned with the provider-specific
			implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return [ 'file_search_call.results', 'web_search_call.results',
		         'web_search_call.action.sources', 'code_interpreter_call.outputs',
		         'reasoning.encrypted_content', 'message.output_text.logprobs', ]
	
	@property
	def tool_options( self ) -> List[ str ] | None:
		"""Get tool options.
		
		Purpose:
			Returns the tool options exposed by the Chat wrapper. The property centralizes UI option
			values and keeps application selectors aligned with the provider-specific
			implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return [ 'web_search', 'file_search', ]
	
	@property
	def choice_options( self ) -> List[ str ] | None:
		"""Get choice options.
		
		Purpose:
			Returns the choice options exposed by the Chat wrapper. The property centralizes UI
			option values and keeps application selectors aligned with the provider-specific
			implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return [ 'auto', 'required', 'none', ]
	
	@property
	def purpose_options( self ) -> List[ str ] | None:
		"""Get purpose options.
		
		Purpose:
			Returns the purpose options exposed by the Chat wrapper. The property centralizes UI
			option values and keeps application selectors aligned with the provider-specific
			implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return [ 'assistants', 'batch', 'fine-tune', 'vision', 'user_data', 'evals', ]
	
	@property
	def format_options( self ) -> List[ str ] | None:
		"""Get format options.
		
		Purpose:
			Returns the format options exposed by the Chat wrapper. The property centralizes UI
			option values and keeps application selectors aligned with the provider-specific
			implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return [ 'text', 'json_object', 'json_schema', ]
	
	@property
	def reasoning_options( self ) -> List[ str ] | None:
		"""Get reasoning options.
		
		Purpose:
			Returns the reasoning options exposed by the Chat wrapper. The property centralizes UI
			option values and keeps application selectors aligned with the provider-specific
			implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return [ 'none', 'minimal', 'low', 'medium', 'high', 'xhigh', ]
	
	@property
	def modality_options( self ) -> List[ str ] | None:
		"""Get modality options.
		
		Purpose:
			Returns the modality options exposed by the Chat wrapper. The property centralizes UI
			option values and keeps application selectors aligned with the provider-specific
			implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return [ 'text', ]
	
	def supports_reasoning_model( self, model: str = None ) -> bool:
		"""Supports reasoning model.
		
		Purpose:
			Executes the supports reasoning model operation for the Chat wrapper. The method
			validates required inputs, updates runtime state, and returns the application-facing
			result produced by the operation.
		
		Args:
			model (str): Model value used by the operation.
		
		Returns:
			Application-facing result produced by the operation.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			model_value = model if isinstance( model, str ) else self.model
			if not isinstance( model_value, str ) or not model_value.strip( ):
				return False
			
			name = model_value.strip( ).lower( )
			return name.startswith( 'gpt-5' ) or name.startswith( 'o' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'supports_reasoning_model( self, model )'
			Logger( ).write( exception )
			raise exception
	
	def normalize_reasoning_effort( self, reasoning: str | Dict[ str, str ] = None,
			model: str = None ) -> str | None:
		"""Normalize reasoning effort.
		
		Purpose:
			Normalizes the reasoning effort value used for the Chat workflow. The method converts
			provider-specific objects, dictionaries, or compatibility inputs into a stable structure
			for application use.
		
		Args:
			reasoning (str | Dict[str, str]): Reasoning value used by the operation.
			model (str): Model value used by the operation.
		
		Returns:
			Normalized application-facing value or structure.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			if reasoning is None:
				return None
			
			if not self.supports_reasoning_model( model ):
				return None
			
			if isinstance( reasoning, dict ):
				value = reasoning.get( 'effort' )
			else:
				value = reasoning
			
			if not isinstance( value, str ) or not value.strip( ):
				return None
			
			effort = value.strip( ).lower( )
			if effort == 'none':
				return None
			
			if effort not in self.reasoning_options:
				return None
			
			model_value = model if isinstance( model, str ) else self.model
			model_name = str( model_value or '' ).strip( ).lower( )
			
			if model_name.startswith( 'gpt-5.1' ) and effort in [ 'minimal', 'xhigh' ]:
				return None
			
			if model_name.startswith( 'gpt-5-pro' ):
				return 'high'
			
			if effort == 'xhigh' and not (
					model_name.startswith( 'gpt-5.4' ) or model_name.startswith( 'gpt-5.5' )):
				return 'high'
			
			return effort
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'normalize_reasoning_effort( self, reasoning, model )'
			Logger( ).write( exception )
			raise exception
	
	def build_reasoning( self, reasoning: str | Dict[ str, str ] = None,
			model: str = None ) -> Dict[ str, str ] | None:
		"""Build reasoning.
		
		Purpose:
			Builds the reasoning payload used for the Chat workflow. The method validates caller
			input, applies compatibility defaults, and returns a provider-ready structure without
			executing the provider request.
		
		Args:
			reasoning (str | Dict[str, str]): Reasoning value used by the operation.
			model (str): Model value used by the operation.
		
		Returns:
			Provider-ready request structure or omitted optional payload.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			effort = self.normalize_reasoning_effort( reasoning=reasoning, model=model )
			return { 'effort': effort } if effort else None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'build_reasoning( self, reasoning, model )'
			Logger( ).write( exception )
			raise exception
	
	def normalize_max_output_tokens( self, max_tokens: int = None,
			model: str = None ) -> int | None:
		"""Normalize max output tokens.
		
		Purpose:
			Normalizes the max output tokens value used for the Chat workflow. The method converts
			provider-specific objects, dictionaries, or compatibility inputs into a stable structure
			for application use.
		
		Args:
			max_tokens (int): Max tokens value used by the operation.
			model (str): Model value used by the operation.
		
		Returns:
			Normalized application-facing value or structure.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			if max_tokens is None:
				return None
			
			try:
				value = int( max_tokens )
			except Exception as e:
				exception = Error( e )
				exception.module = 'gpt'
				exception.cause = 'Chat'
				exception.method = 'normalize_max_output_tokens( ... )'
				Logger( ).write( exception )
				return None
			
			if value <= 0:
				return None
			
			model_value = model if isinstance( model, str ) else self.model
			model_name = str( model_value or '' ).strip( ).lower( )
			limit = 16384
			
			if model_name.startswith( 'gpt-5' ):
				limit = 32768
			elif model_name.startswith( 'gpt-4.1' ):
				limit = 32768
			elif model_name.startswith( 'o' ):
				limit = 32768
			
			return min( value, limit )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'normalize_max_output_tokens( self, max_tokens, model )'
			Logger( ).write( exception )
			raise exception
	
	def normalize_domains( self, allowed_domains: List[ str ] = None ) -> List[ str ]:
		"""Normalize domains.
		
		Purpose:
			Normalizes the domains value used for the Chat workflow. The method converts
			provider-specific objects, dictionaries, or compatibility inputs into a stable structure
			for application use.
		
		Args:
			allowed_domains (List[str]): Allowed domains value used by the operation.
		
		Returns:
			Normalized application-facing value or structure.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			if allowed_domains is None:
				return [ ]
			
			domains: List[ str ] = [ ]
			for domain in allowed_domains:
				if not isinstance( domain, str ) or not domain.strip( ):
					continue
				
				value = domain.strip( ).lower( )
				value = value.replace( 'https://', '' ).replace( 'http://', '' )
				value = value.split( '/' )[ 0 ].strip( )
				
				if value and value not in domains:
					domains.append( value )
			
			return domains
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'normalize_domains( self, allowed_domains )'
			Logger( ).write( exception )
			raise exception
	
	def build_input( self, prompt: str, context: List[ Dict[ str, str ] ] = None,
			input_data: List[ Dict[ str, Any ] ] = None ) -> List[ Dict[ str, Any ] ]:
		"""Build input.
		
		Purpose:
			Builds the input payload used for the Chat workflow. The method validates caller input,
			applies compatibility defaults, and returns a provider-ready structure without executing
			the provider request.
		
		Args:
			prompt (str): Prompt value used by the operation.
			context (List[Dict[str, str]]): Context value used by the operation.
			input_data (List[Dict[str, Any]]): Input data value used by the operation.
		
		Returns:
			Provider-ready request structure or omitted optional payload.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			throw_if( 'prompt', prompt )
			self.messages = [ ]
			
			if input_data is not None and len( input_data ) > 0:
				self.messages.extend( input_data )
			elif context is not None and len( context ) > 0:
				for item in context:
					if not isinstance( item, dict ):
						continue
					
					role = str( item.get( 'role', '' ) or '' ).strip( )
					content = item.get( 'content', '' )
					
					if role not in [ 'user', 'assistant', 'system', 'developer' ]:
						continue
					
					if not isinstance( content, str ) or not content.strip( ):
						continue
					
					self.messages.append(
						{
								'role': role,
								'content': [
										{
												'type': 'input_text',
												'text': content.strip( ),
										}, ],
						} )
			
			self.messages.append(
				{
						'role': 'user',
						'content': [
								{
										'type': 'input_text',
										'text': prompt,
								}, ],
				} )
			
			return self.messages
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'build_input( self, prompt, context, input_data )'
			Logger( ).write( exception )
			raise exception
	
	def build_tools( self, tools: List[ Dict[ str, Any ] ] = None,
			allowed_domains: List[ str ] = None,
			vector_store_ids: List[ str ] = None ) -> List[ Dict[ str, Any ] ] | None:
		"""Build tools.
		
		Purpose:
			Builds the tools payload used for the Chat workflow. The method validates caller input,
			applies compatibility defaults, and returns a provider-ready structure without executing
			the provider request.
		
		Args:
			tools (List[Dict[str, Any]]): Tools value used by the operation.
			allowed_domains (List[str]): Allowed domains value used by the operation.
			vector_store_ids (List[str]): Vector store ids value used by the operation.
		
		Returns:
			Provider-ready request structure or omitted optional payload.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			self.allowed_domains = self.normalize_domains( allowed_domains )
			self.vector_store_ids = vector_store_ids if vector_store_ids is not None else [ ]
			if tools is None or len( tools ) == 0:
				return None
			
			self.built_tools = [ ]
			for tool in tools:
				if not isinstance( tool, dict ):
					continue
				
				tool_type = str( tool.get( 'type', '' ) or '' ).strip( )
				if not tool_type:
					continue
				
				if tool_type in [ 'web_search', 'web_search_preview',
				                  'web_search_preview_2025_03_11' ]:
					built_tool: Dict[ str, Any ] = { 'type': 'web_search' }
					if len( self.allowed_domains ) > 0:
						built_tool[ 'filters' ] = { 'allowed_domains': self.allowed_domains }
					
					self.built_tools.append( built_tool )
					continue
				
				if tool_type == 'file_search':
					if len( self.vector_store_ids ) == 0:
						continue
					
					self.built_tools.append( {
							'type': 'file_search',
							'vector_store_ids': self.vector_store_ids,
					} )
					continue
			return self.built_tools if len( self.built_tools ) > 0 else None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'build_tools( self, tools, allowed_domains, vector_store_ids )'
			Logger( ).write( exception )
			raise exception
	
	def build_tool_choice( self, tool_choice: str = None,
			tools: List[ Dict[ str, Any ] ] = None ) -> str | None:
		"""Build tool choice.
		
		Purpose:
			Builds the tool choice payload used for the Chat workflow. The method validates caller
			input, applies compatibility defaults, and returns a provider-ready structure without
			executing the provider request.
		
		Args:
			tool_choice (str): Tool choice value used by the operation.
			tools (List[Dict[str, Any]]): Tools value used by the operation.
		
		Returns:
			Provider-ready request structure or omitted optional payload.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			if not isinstance( tool_choice, str ) or not tool_choice.strip( ):
				return None
			
			choice = tool_choice.strip( )
			if choice not in self.choice_options:
				return None
			
			if choice == 'none':
				return 'none'
			
			if tools is None or len( tools ) == 0:
				return None
			
			return choice
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'build_tool_choice( self, tool_choice, tools )'
			Logger( ).write( exception )
			raise exception
	
	def build_include( self, include: List[ str ] = None,
			tools: List[ Dict[ str, Any ] ] = None ) -> List[ str ] | None:
		"""Build include.
		
		Purpose:
			Builds the include payload used for the Chat workflow. The method validates caller
			input, applies compatibility defaults, and returns a provider-ready structure without
			executing the provider request.
		
		Args:
			include (List[str]): Include value used by the operation.
			tools (List[Dict[str, Any]]): Tools value used by the operation.
		
		Returns:
			Provider-ready request structure or omitted optional payload.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			if include is None or len( include ) == 0:
				return None
			
			tool_types = [ ]
			if isinstance( tools, list ):
				for tool in tools:
					if isinstance( tool, dict ) and tool.get( 'type' ):
						tool_types.append( str( tool.get( 'type' ) ) )
			
			allowed = [ ]
			for value in include:
				if not isinstance( value, str ) or not value.strip( ):
					continue
				
				name = value.strip( )
				if name == 'reasoning.encrypted_content' and self.reasoning is not None:
					allowed.append( name )
					continue
				
				if name == 'message.output_text.logprobs':
					allowed.append( name )
					continue
				
				if name.startswith( 'web_search_call.' ) and 'web_search' in tool_types:
					allowed.append( name )
					continue
				
				if name == 'file_search_call.results' and 'file_search' in tool_types:
					allowed.append( name )
					continue
			
			return allowed if len( allowed ) > 0 else None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'build_include( self, include, tools )'
			Logger( ).write( exception )
			raise exception
	
	def build_text_format( self, format: Dict[ str, Any ] | str = None ) -> Dict[ str, Any ] | None:
		"""Build text format.
		
		Purpose:
			Builds the text format payload used for the Chat workflow. The method validates caller
			input, applies compatibility defaults, and returns a provider-ready structure without
			executing the provider request.
		
		Args:
			format (Dict[str, Any] | str): Format value used by the operation.
		
		Returns:
			Provider-ready request structure or omitted optional payload.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			if format is None:
				return None
			
			if isinstance( format, dict ) and len( format ) > 0:
				if 'format' in format and isinstance( format.get( 'format' ), dict ):
					return format
				
				if 'type' in format:
					return { 'format': format }
				
				return None
			
			if isinstance( format, str ) and format.strip( ):
				value = format.strip( )
				if value == 'text':
					return { 'format': { 'type': 'text' } }
				
				if value == 'json_object':
					return { 'format': { 'type': 'json_object' } }
			
			return None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'build_text_format( self, format )'
			Logger( ).write( exception )
			raise exception
	
	def build_request( self, prompt: str, model: str, temperature: float = None,
			format: Dict[ str, Any ] = None, top_p: float = None, frequency: float = None,
			max_tools: int = None, presence: float = None, max_tokens: int = None,
			store: bool = None, stream: bool = None, instruct: str = None,
			background: bool = False, reasoning: str = None, include: List[ str ] = None,
			tools: List[ Dict[ str, Any ] ] = None, allowed_domains: List[ str ] = None,
			previous_id: str = None, tool_choice: str = None, is_parallel: bool = None,
			context: List[ Dict[ str, str ] ] = None, input_data: List[ Dict[ str, Any ] ] = None,
			vector_store_ids: List[ str ] = None, conversation_id: str = None ) -> Dict[ str, Any ]:
		"""Build request.
		
		Purpose:
			Builds the request payload used for the Chat workflow. The method validates caller
			input, applies compatibility defaults, and returns a provider-ready structure without
			executing the provider request.
		
		Args:
			prompt (str): Prompt value used by the operation.
			model (str): Model value used by the operation.
			temperature (float): Temperature value used by the operation.
			format (Dict[str, Any]): Format value used by the operation.
			top_p (float): Top p value used by the operation.
			frequency (float): Frequency value used by the operation.
			max_tools (int): Max tools value used by the operation.
			presence (float): Presence value used by the operation.
			max_tokens (int): Max tokens value used by the operation.
			store (bool): Store value used by the operation.
			stream (bool): Stream value used by the operation.
			instruct (str): Instruct value used by the operation.
			background (bool): Background value used by the operation.
			reasoning (str): Reasoning value used by the operation.
			include (List[str]): Include value used by the operation.
			tools (List[Dict[str, Any]]): Tools value used by the operation.
			allowed_domains (List[str]): Allowed domains value used by the operation.
			previous_id (str): Previous id value used by the operation.
			tool_choice (str): Tool choice value used by the operation.
			is_parallel (bool): Is parallel value used by the operation.
			context (List[Dict[str, str]]): Context value used by the operation.
			input_data (List[Dict[str, Any]]): Input data value used by the operation.
			vector_store_ids (List[str]): Vector store ids value used by the operation.
			conversation_id (str): Conversation id value used by the operation.
		
		Returns:
			Provider-ready request structure or omitted optional payload.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'model', model )
			self.model = model.strip( ) if isinstance( model, str ) else model
			self.prompt = prompt
			self.temperature = temperature
			self.top_percent = top_p
			self.frequency_penalty = frequency
			self.presence_penalty = presence
			self.max_tokens = self.normalize_max_output_tokens( max_tokens=max_tokens,
				model=self.model )
			self.store = store
			self.stream = stream
			self.background = background
			self.instructions = instruct
			self.response_format = self.build_text_format( format )
			self.max_tools = max_tools
			self.vector_store_ids = vector_store_ids if vector_store_ids is not None else [ ]
			self.previous_id = previous_id if isinstance( previous_id, str ) else None
			self.conversation_id = conversation_id if isinstance( conversation_id, str ) else None
			self.parallel_tools = is_parallel
			self.reasoning = self.build_reasoning( reasoning=reasoning, model=self.model )
			self.tools = self.build_tools( tools=tools, allowed_domains=allowed_domains,
				vector_store_ids=self.vector_store_ids )
			self.tool_choice = self.build_tool_choice( tool_choice=tool_choice, tools=self.tools )
			self.include = self.build_include( include=include, tools=self.tools )
			self.effective_context = [ ]
			if not (self.conversation_id and self.conversation_id.strip( )):
				self.effective_context = context
			
			self.input = self.build_input( prompt=prompt, context=self.effective_context,
				input_data=input_data )
			self.request = {
					'model': self.model,
					'input': self.input,
			}
			
			if self.instructions:
				self.request[ 'instructions' ] = self.instructions
			
			if self.reasoning is not None:
				self.request[ 'reasoning' ] = self.reasoning
			
			if isinstance( self.max_tokens, int ) and self.max_tokens > 0:
				self.request[ 'max_output_tokens' ] = self.max_tokens
			
			if self.temperature is not None and not self.model.startswith( 'gpt-5' ):
				self.request[ 'temperature' ] = self.temperature
			
			if self.top_percent is not None and not self.model.startswith( 'gpt-5' ):
				self.request[ 'top_p' ] = self.top_percent
			
			if self.frequency_penalty is not None and not self.model.startswith( 'gpt-5' ):
				self.request[ 'frequency_penalty' ] = self.frequency_penalty
			
			if self.presence_penalty is not None and not self.model.startswith( 'gpt-5' ):
				self.request[ 'presence_penalty' ] = self.presence_penalty
			
			if self.store is not None:
				self.request[ 'store' ] = self.store
			
			if self.include is not None and len( self.include ) > 0:
				self.request[ 'include' ] = self.include
			
			if self.tools is not None and len( self.tools ) > 0:
				self.request[ 'tools' ] = self.tools
			
			if self.tool_choice:
				self.request[ 'tool_choice' ] = self.tool_choice
			
			if self.parallel_tools is not None and self.tools is not None:
				self.request[ 'parallel_tool_calls' ] = self.parallel_tools
			
			if self.previous_id and self.previous_id.strip( ):
				self.request[ 'previous_response_id' ] = self.previous_id.strip( )
			
			if self.conversation_id and self.conversation_id.strip( ):
				self.request[ 'conversation' ] = self.conversation_id.strip( )
			
			if isinstance( self.max_tools, int ) and self.max_tools > 0 and self.tools is not None:
				self.request[ 'max_tool_calls' ] = self.max_tools
			
			if self.response_format is not None and len( self.response_format ) > 0:
				self.request[ 'text' ] = self.response_format
			
			return self.request
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'build_request( self, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def get_output_text( self ) -> str | None:
		"""Get output text.
		
		Purpose:
			Returns the output text value for the active Chat request. The method inspects current
			runtime state and provides a safe application-facing result.
		
		Returns:
			Requested value derived from the current runtime state.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			if self.response is None:
				return None
			
			self.output_text = getattr( self.response, 'output_text', None )
			if self.output_text:
				return self.output_text
			
			if hasattr( self.response, 'output' ) and self.response.output:
				text_parts = [ ]
				for item in self.response.output:
					if getattr( item, 'type', None ) != 'message':
						continue
					
					if not hasattr( item, 'content' ) or item.content is None:
						continue
					
					for block in item.content:
						if getattr( block, 'type', None ) == 'output_text':
							text = getattr( block, 'text', None )
							if text:
								text_parts.append( text )
				
				if len( text_parts ) > 0:
					self.output_text = ''.join( text_parts ).strip( )
					return self.output_text
			
			return None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'get_output_text( self ) -> str | None'
			Logger( ).write( exception )
			raise exception
	
	def get_usage( self ) -> Any:
		"""Get usage.
		
		Purpose:
			Returns the usage value for the active Chat request. The method inspects current runtime
			state and provides a safe application-facing result.
		
		Returns:
			Requested value derived from the current runtime state.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			if self.response is None:
				return None
			
			return getattr( self.response, 'usage', None )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'get_usage( self ) -> Any'
			Logger( ).write( exception )
			raise exception
	
	def generate_text( self, prompt: str, model: str, temperature: float = None,
			format: Dict[ str, Any ] = None, top_p: float = None, frequency: float = None,
			max_tools: int = None, presence: float = None, max_tokens: int = None,
			store: bool = None, stream: bool = None, instruct: str = None, background: bool = False,
			reasoning: str = None, include: List[ str ] = None,
			tools: List[ Dict[ str, Any ] ] = None,
			allowed_domains: List[ str ] = None, previous_id: str = None, tool_choice: str = None,
			is_parallel: bool = None, context: List[ Dict[ str, str ] ] = None,
			input_data: List[ Dict[ str, Any ] ] = None, vector_store_ids: List[ str ] = None,
			conversation_id: str = None ) -> str | None:
		"""Generate text.
		
		Purpose:
			Generates provider output for the Chat workflow using validated model settings and
			request inputs. The method coordinates request construction, provider execution,
			response capture, and logged exception handling.
		
		Args:
			prompt (str): Prompt value used by the operation.
			model (str): Model value used by the operation.
			temperature (float): Temperature value used by the operation.
			format (Dict[str, Any]): Format value used by the operation.
			top_p (float): Top p value used by the operation.
			frequency (float): Frequency value used by the operation.
			max_tools (int): Max tools value used by the operation.
			presence (float): Presence value used by the operation.
			max_tokens (int): Max tokens value used by the operation.
			store (bool): Store value used by the operation.
			stream (bool): Stream value used by the operation.
			instruct (str): Instruct value used by the operation.
			background (bool): Background value used by the operation.
			reasoning (str): Reasoning value used by the operation.
			include (List[str]): Include value used by the operation.
			tools (List[Dict[str, Any]]): Tools value used by the operation.
			allowed_domains (List[str]): Allowed domains value used by the operation.
			previous_id (str): Previous id value used by the operation.
			tool_choice (str): Tool choice value used by the operation.
			is_parallel (bool): Is parallel value used by the operation.
			context (List[Dict[str, str]]): Context value used by the operation.
			input_data (List[Dict[str, Any]]): Input data value used by the operation.
			vector_store_ids (List[str]): Vector store ids value used by the operation.
			conversation_id (str): Conversation id value used by the operation.
		
		Returns:
			Generated or extracted text when available.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'model', model )
			self.client = OpenAI( api_key=self.api_key )
			self.stream_requested = bool( stream )
			self.background_requested = bool( background )
			self.request = self.build_request( prompt=prompt, model=model,
				temperature=temperature, format=format, top_p=top_p, frequency=frequency,
				max_tools=max_tools, presence=presence, max_tokens=max_tokens, store=store,
				stream=False, instruct=instruct, background=False, reasoning=reasoning,
				include=include, tools=tools, allowed_domains=allowed_domains,
				previous_id=previous_id, tool_choice=tool_choice, is_parallel=is_parallel,
				context=context, input_data=input_data, vector_store_ids=vector_store_ids,
				conversation_id=conversation_id )
			
			self.response = self.client.responses.create( **self.request )
			self.previous_id = getattr( self.response, 'id', None )
			self.output_text = self.get_output_text( )
			return self.output_text
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Chat'
			exception.method = 'generate_text( self, prompt: str ) -> str | None'
			Logger( ).write( exception )
			raise exception
	
	def __dir__( self ) -> List[ str ] | None:
		"""Return member names.
		
		Purpose:
			Returns a stable list of public members exposed by the Chat object for interactive
			inspection, debugging, and application-level compatibility.
		
		Returns:
			Member names exposed for inspection.
		"""
		return [
				'api_key',
				'client',
				'model',
				'prompt',
				'temperature',
				'top_percent',
				'frequency_penalty',
				'presence_penalty',
				'max_tokens',
				'stops',
				'store',
				'stream',
				'background',
				'number',
				'response_format',
				'context',
				'instructions',
				'include',
				'tool_choice',
				'previous_id',
				'conversation_id',
				'parallel_tools',
				'max_tools',
				'input',
				'tools',
				'reasoning',
				'allowed_domains',
				'max_search_results',
				'output_text',
				'vector_store_ids',
				'file_ids',
				'response',
				'file',
				'purpose',
				'model_options',
				'include_options',
				'tool_options',
				'choice_options',
				'purpose_options',
				'format_options',
				'reasoning_options',
				'modality_options',
				'supports_reasoning_model',
				'normalize_reasoning_effort',
				'normalize_max_output_tokens',
				'normalize_domains',
				'effective_context',
				'build_reasoning',
				'build_input',
				'build_tools',
				'build_tool_choice',
				'build_include',
				'build_text_format',
				'build_request',
				'get_output_text',
				'get_usage',
				'generate_text',
		]

class Images( GPT ):
	"""Provide Images workflow support.
	
	Purpose:
		Provides OpenAI image generation, image editing, and image analysis workflows. The class
		stores image request options, uploaded file references, output format settings, and
		response data used by Gipity Image mode.
	
	Attributes:
		quality (Optional[str]): Quality retained by the provider wrapper.
		detail (Optional[str]): Detail retained by the provider wrapper.
		size (Optional[str]): Size retained by the provider wrapper.
		previous_id (Optional[str]): Previous id retained by the provider wrapper.
		include (Optional[List[str]]): Include retained by the provider wrapper.
		tool_choice (Optional[str]): Tool choice retained by the provider wrapper.
		parallel_tools (Optional[bool]): Parallel tools retained by the provider wrapper.
		input (Optional[List[Dict[str, str]] | str]): Input retained by the provider wrapper.
		instructions (Optional[str]): Instructions retained by the provider wrapper.
		max_tools (Optional[int]): Max tools retained by the provider wrapper.
		tools (Optional[List[Dict[str, str]]]): Tools retained by the provider wrapper.
		messages (Optional[List[Dict[str, str]]]): Messages retained by the provider wrapper.
		reasoning (Optional[Dict[str, str]]): Reasoning retained by the provider wrapper.
		image_url (Optional[str]): Image url retained by the provider wrapper.
		image_path (Optional[str]): Image path retained by the provider wrapper.
		file_url (Optional[str]): File url retained by the provider wrapper.
		file_path (Optional[str]): File path retained by the provider wrapper.
		style (Optional[str]): Style retained by the provider wrapper.
		allowed_domains (Optional[List[str]]): Allowed domains retained by the provider wrapper.
		response_format (Optional[str]): Response format retained by the provider wrapper.
		mime_format (Optional[str]): Mime format retained by the provider wrapper.
		background (Optional[bool]): Background retained by the provider wrapper.
		backcolor (Optional[str]): Backcolor retained by the provider wrapper.
		compression (Optional[float]): Compression retained by the provider wrapper.
	"""
	quality: Optional[ str ]
	detail: Optional[ str ]
	size: Optional[ str ]
	previous_id: Optional[ str ]
	include: Optional[ List[ str ] ]
	tool_choice: Optional[ str ]
	parallel_tools: Optional[ bool ]
	input: Optional[ List[ Dict[ str, str ] ] | str ]
	instructions: Optional[ str ]
	max_tools: Optional[ int ]
	tools: Optional[ List[ Dict[ str, str ] ] ]
	messages: Optional[ List[ Dict[ str, str ] ] ]
	reasoning: Optional[ Dict[ str, str ] ]
	image_url: Optional[ str ]
	image_path: Optional[ str ]
	file_url: Optional[ str ]
	file_path: Optional[ str ]
	style: Optional[ str ]
	allowed_domains: Optional[ List[ str ] ]
	response_format: Optional[ str ]
	mime_format: Optional[ str ]
	background: Optional[ bool ]
	backcolor: Optional[ str ]
	compression: Optional[ float ]
	
	def __init__( self, prompt: str = None, model: str = 'gpt-image-1', temperature: float = None,
			top_p: float = None, presence: float = None, frequency: float = None,
			max_tokens: int = None, store: bool = None, stream: bool = False, backcolor: str = None,
			instruct: str = None, background: bool = None, number: int = None,
			image_format: str = None, include: List[ Dict[ str, str ] ] = None,
			tools: List[ Dict[ str, str ] ] = None, max_tools: int = None,
			respose_format: Dict[ str, str ] = None, response_format: Dict[ str, str ] = None,
			tool_choice: str = None, image_path: str = None, is_parallel: bool = None,
			input: List[ Dict[ str, str ] ] = None, previous_id: str = None,
			reasoning: Dict[ str, str ] = None, input_text: str = None, image_url: str = None,
			content: List[ Dict[ str, str ] ] = None, quality: str = None, size: str = None,
			detail: str = None, style: str = None, compression: float = None ):
		"""Initialize instance.
		
		Purpose:
			Initializes the Images object with default configuration, runtime state, provider
			settings, and compatibility fields. This constructor prepares the instance for later
			method calls without performing external work beyond local attribute assignment.
		
		Args:
			prompt (str): Prompt value used by the operation.
			model (str): Model value used by the operation.
			temperature (float): Temperature value used by the operation.
			top_p (float): Top p value used by the operation.
			presence (float): Presence value used by the operation.
			frequency (float): Frequency value used by the operation.
			max_tokens (int): Max tokens value used by the operation.
			store (bool): Store value used by the operation.
			stream (bool): Stream value used by the operation.
			backcolor (str): Backcolor value used by the operation.
			instruct (str): Instruct value used by the operation.
			background (bool): Background value used by the operation.
			number (int): Number value used by the operation.
			image_format (str): Image format value used by the operation.
			include (List[Dict[str, str]]): Include value used by the operation.
			tools (List[Dict[str, str]]): Tools value used by the operation.
			max_tools (int): Max tools value used by the operation.
			respose_format (Dict[str, str]): Respose format value used by the operation.
			response_format (Dict[str, str]): Response format value used by the operation.
			tool_choice (str): Tool choice value used by the operation.
			image_path (str): Image path value used by the operation.
			is_parallel (bool): Is parallel value used by the operation.
			input (List[Dict[str, str]]): Input value used by the operation.
			previous_id (str): Previous id value used by the operation.
			reasoning (Dict[str, str]): Reasoning value used by the operation.
			input_text (str): Input text value used by the operation.
			image_url (str): Image url value used by the operation.
			content (List[Dict[str, str]]): Content value used by the operation.
			quality (str): Quality value used by the operation.
			size (str): Size value used by the operation.
			detail (str): Detail value used by the operation.
			style (str): Style value used by the operation.
			compression (float): Compression value used by the operation.
		"""
		super( ).__init__( )
		self.api_key = cfg.OPENAI_API_KEY
		self.client = None
		self.model = model
		self.prompt = prompt
		self.number = number
		self.previous_id = previous_id
		self.temperature = temperature
		self.top_percent = top_p
		self.frequency_penalty = frequency
		self.presence_penalty = presence
		self.max_tokens = max_tokens
		self.store = store
		self.stream = stream
		self.instruct = instruct
		self.instructions = instruct
		self.max_tools = max_tools
		self.reasoning = reasoning
		self.tools = tools
		self.tool_choice = tool_choice
		self.input_text = input_text if input_text is not None else prompt
		self.input = input
		self.content = content
		self.background = background
		self.backcolor = backcolor
		self.image_path = image_path
		self.image_url = image_url
		self.include = include
		self.quality = quality
		self.detail = detail
		self.size = size
		self.style = style
		self.compression = compression
		self.response_format = response_format if response_format is not None else respose_format
		self.mime_format = image_format
		self.parallel_tools = is_parallel
		self.response = None
		self.file = None
		self.request = { }
		self.data = None
		self.outputs = [ ]
		self.output_text = None
		self.output_format = None
		self.output_compression = None
		self.image_content = None
	
	@property
	def style_options( self ) -> List[ str ]:
		"""Get style options.
		
		Purpose:
			Returns the style options exposed by the Images wrapper. The property centralizes UI
			option values and keeps application selectors aligned with the provider-specific
			implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return [
				'vivid',
				'natural',
		]
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Get model options.
		
		Purpose:
			Returns the model options exposed by the Images wrapper. The property centralizes UI
			option values and keeps application selectors aligned with the provider-specific
			implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return [
				'gpt-image-2',
				'gpt-image-1.5',
				'gpt-image-1',
				'gpt-image-1-mini',
		]
	
	@property
	def size_options( self ) -> List[ str ]:
		"""Get size options.
		
		Purpose:
			Returns the size options exposed by the Images wrapper. The property centralizes UI
			option values and keeps application selectors aligned with the provider-specific
			implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return [
				'auto',
				'1024x1024',
				'1024x1536',
				'1536x1024',
		]
	
	@property
	def analysis_model_options( self ) -> List[ str ] | None:
		"""Get analysis model options.
		
		Purpose:
			Returns the analysis model options exposed by the Images wrapper. The property
			centralizes UI option values and keeps application selectors aligned with the
			provider-specific implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return [
				'gpt-5.4',
				'gpt-5.4-mini',
				'gpt-5',
				'gpt-5-mini',
				'gpt-4.1',
				'gpt-4.1-mini',
				'gpt-4o',
				'gpt-4o-mini',
		]
	
	@property
	def format_options( self ) -> List[ str ]:
		"""Get format options.
		
		Purpose:
			Returns the format options exposed by the Images wrapper. The property centralizes UI
			option values and keeps application selectors aligned with the provider-specific
			implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return [
				'url',
				'b64_json',
		]
	
	@property
	def mime_options( self ) -> List[ str ]:
		"""Get mime options.
		
		Purpose:
			Returns the mime options exposed by the Images wrapper. The property centralizes UI
			option values and keeps application selectors aligned with the provider-specific
			implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return [
				'png',
				'jpeg',
				'webp',
		]
	
	@property
	def include_options( self ) -> List[ str ] | None:
		"""Get include options.
		
		Purpose:
			Returns the include options exposed by the Images wrapper. The property centralizes UI
			option values and keeps application selectors aligned with the provider-specific
			implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return [
				'file_search_call.results',
				'web_search_call.results',
				'web_search_call.action.sources',
				'message.input_image.image_url',
				'computer_call_output.output.image_url',
				'code_interpreter_call.outputs',
				'reasoning.encrypted_content',
				'message.output_text.logprobs',
		]
	
	@property
	def tool_options( self ) -> List[ str ] | None:
		"""Get tool options.
		
		Purpose:
			Returns the tool options exposed by the Images wrapper. The property centralizes UI
			option values and keeps application selectors aligned with the provider-specific
			implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return [
				'web_search',
				'image_generation',
				'file_search',
				'code_interpreter',
				'computer_use_preview',
		]
	
	@property
	def choice_options( self ) -> List[ str ] | None:
		"""Get choice options.
		
		Purpose:
			Returns the choice options exposed by the Images wrapper. The property centralizes UI
			option values and keeps application selectors aligned with the provider-specific
			implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return [
				'auto',
				'required',
				'none',
		]
	
	@property
	def backcolor_options( self ) -> List[ str ]:
		"""Get backcolor options.
		
		Purpose:
			Returns the backcolor options exposed by the Images wrapper. The property centralizes UI
			option values and keeps application selectors aligned with the provider-specific
			implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return [
				'auto',
				'transparent',
				'opaque',
		]
	
	@property
	def quality_options( self ) -> List[ str ]:
		"""Get quality options.
		
		Purpose:
			Returns the quality options exposed by the Images wrapper. The property centralizes UI
			option values and keeps application selectors aligned with the provider-specific
			implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return [
				'auto',
				'low',
				'medium',
				'high',
		]
	
	@property
	def detail_options( self ) -> List[ str ]:
		"""Get detail options.
		
		Purpose:
			Returns the detail options exposed by the Images wrapper. The property centralizes UI
			option values and keeps application selectors aligned with the provider-specific
			implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return [
				'auto',
				'low',
				'high',
				'original',
		]
	
	@property
	def reasoning_options( self ) -> List[ str ] | None:
		"""Get reasoning options.
		
		Purpose:
			Returns the reasoning options exposed by the Images wrapper. The property centralizes UI
			option values and keeps application selectors aligned with the provider-specific
			implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return [
				'low',
				'medium',
				'high',
				'none',
				'minimal',
				'xhigh',
		]
	
	@property
	def modality_options( self ) -> List[ str ] | None:
		"""Get modality options.
		
		Purpose:
			Returns the modality options exposed by the Images wrapper. The property centralizes UI
			option values and keeps application selectors aligned with the provider-specific
			implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return [
				'text',
				'auto',
				'image',
				'audio',
		]
	
	def supports_original_detail( self ) -> bool:
		"""Supports original detail.
		
		Purpose:
			Executes the supports original detail operation for the Images wrapper. The method
			validates required inputs, updates runtime state, and returns the application-facing
			result produced by the operation.
		
		Returns:
			Application-facing result produced by the operation.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			model_name = str( self.model or '' ).strip( ).lower( )
			return model_name.startswith( 'gpt-5.4' ) or model_name.startswith( 'gpt-5.5' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Images'
			exception.method = 'supports_original_detail( self )'
			Logger( ).write( exception )
			raise exception
	
	def get_analysis_detail( self ) -> str:
		"""Get analysis detail.
		
		Purpose:
			Returns the analysis detail value for the active Images request. The method inspects
			current runtime state and provides a safe application-facing result.
		
		Returns:
			Requested value derived from the current runtime state.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			if not isinstance( self.detail, str ) or not self.detail.strip( ):
				return 'auto'
			
			if self.detail == 'original' and not self.supports_original_detail( ):
				return 'high'
			
			return self.detail
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Images'
			exception.method = 'get_analysis_detail( self )'
			Logger( ).write( exception )
			raise exception
	
	def get_output_compression( self ) -> int | None:
		"""Get output compression.
		
		Purpose:
			Returns the output compression value for the active Images request. The method inspects
			current runtime state and provides a safe application-facing result.
		
		Returns:
			Requested value derived from the current runtime state.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			if self.output_format not in [ 'jpeg', 'webp' ]:
				return None
			
			if self.compression is None:
				return None
			
			value = float( self.compression )
			if value <= 0:
				return None
			
			if value <= 1:
				value *= 100
			
			return max( 1, min( 100, int( round( value ) ) ) )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Images'
			exception.method = 'get_output_compression( self )'
			Logger( ).write( exception )
			raise exception
	
	def extract_image_outputs( self ) -> str | bytes | list[ str | bytes ] | None:
		"""Extract image outputs.
		
		Purpose:
			Executes the extract image outputs operation for the Images wrapper. The method
			validates required inputs, updates runtime state, and returns the application-facing
			result produced by the operation.
		
		Returns:
			Application-facing result produced by the operation.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			self.data = getattr( self.response, 'data', None )
			self.outputs = [ ]
			
			if self.data and len( self.data ) > 0:
				for item in self.data:
					self.b64_json = getattr( item, 'b64_json', None )
					self.url = getattr( item, 'url', None )
					
					if self.b64_json:
						self.outputs.append( base64.b64decode( self.b64_json ) )
					elif self.url:
						self.outputs.append( self.url )
				
				if len( self.outputs ) == 1:
					return self.outputs[ 0 ]
				
				if len( self.outputs ) > 1:
					return self.outputs
			
			return None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Images'
			exception.method = 'extract_image_outputs( self )'
			Logger( ).write( exception )
			raise exception
	
	def get_output_text( self ) -> str | None:
		"""Get output text.
		
		Purpose:
			Returns the output text value for the active Images request. The method inspects current
			runtime state and provides a safe application-facing result.
		
		Returns:
			Requested value derived from the current runtime state.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			if self.response is None:
				return None
			
			self.output_text = getattr( self.response, 'output_text', None )
			if self.output_text:
				return self.output_text
			
			if hasattr( self.response, 'output' ) and self.response.output:
				text_parts = [ ]
				for item in self.response.output:
					if getattr( item, 'type', None ) != 'message':
						continue
					
					if not hasattr( item, 'content' ) or item.content is None:
						continue
					
					for block in item.content:
						if getattr( block, 'type', None ) == 'output_text':
							text = getattr( block, 'text', None )
							if text:
								text_parts.append( text )
				
				if len( text_parts ) > 0:
					self.output_text = ''.join( text_parts ).strip( )
					return self.output_text
			
			return None
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Images'
			exception.method = 'get_output_text( self )'
			Logger( ).write( exception )
			raise exception
	
	def generate( self, prompt: str, number: int = 1, model: str = 'gpt-image-1-mini',
			size: str = '1024x1024', quality: str = 'auto', fmt: str = 'jpeg',
			compression: float = None, background: str = None ) -> str | bytes | list[
		str | bytes ] | None:
		"""Generate.
		
		Purpose:
			Generates one or more images from a text prompt using the configured OpenAI image model.
			The method validates image options, executes the provider request, and returns decoded
			image bytes or URLs.
		
		Args:
			prompt (str): Prompt value used by the operation.
			number (int): Number value used by the operation.
			model (str): Model value used by the operation.
			size (str): Size value used by the operation.
			quality (str): Quality value used by the operation.
			fmt (str): Fmt value used by the operation.
			compression (float): Compression value used by the operation.
			background (str): Background value used by the operation.
		
		Returns:
			Generated image bytes, URLs, multiple outputs, or no value when the provider returns no
			usable output.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'model', model )
			throw_if( 'size', size )
			throw_if( 'quality', quality )
			throw_if( 'fmt', fmt )
			
			self.prompt = prompt
			self.number = number
			self.model = model
			self.size = size
			self.quality = quality
			self.response_format = fmt
			self.output_format = self.response_format.lower( ).replace( '.', '' )
			self.compression = compression
			self.background = background
			self.output_compression = self.get_output_compression( )
			
			if not isinstance( self.number, int ) or self.number <= 0:
				self.number = 1
			
			if self.number > 10:
				self.number = 10
			
			if self.model == 'gpt-image-2' and self.background == 'transparent':
				self.background = 'auto'
			
			self.client = OpenAI( api_key=self.api_key )
			self.request = {
					'model': self.model,
					'prompt': self.prompt,
					'n': self.number,
					'size': self.size,
					'quality': self.quality,
					'output_format': self.output_format,
			}
			
			if self.background:
				self.request[ 'background' ] = self.background
			
			if self.output_compression is not None:
				self.request[ 'output_compression' ] = self.output_compression
			
			self.response = self.client.images.generate( **self.request )
			return self.extract_image_outputs( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Images'
			exception.method = 'generate( self, prompt: str )'
			Logger( ).write( exception )
			raise exception
	
	def analyze( self, text: str, path: str, instruct: str = None, model: str = 'gpt-4o-mini',
			max_tokens: int = None, temperature: float = None, include: List[ str ] = None,
			store: bool = None, stream: bool = None, detail: str = 'auto' ) -> str | None:
		"""Analyze.
		
		Purpose:
			Analyzes a local image with a vision-capable Responses API model. The method uploads the
			image for vision use, builds a multimodal request, and returns the extracted text
			response.
		
		Args:
			text (str): Text value used by the operation.
			path (str): Path value used by the operation.
			instruct (str): Instruct value used by the operation.
			model (str): Model value used by the operation.
			max_tokens (int): Max tokens value used by the operation.
			temperature (float): Temperature value used by the operation.
			include (List[str]): Include value used by the operation.
			store (bool): Store value used by the operation.
			stream (bool): Stream value used by the operation.
			detail (str): Detail value used by the operation.
		
		Returns:
			Application-facing result produced by the operation.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			throw_if( 'text', text )
			throw_if( 'path', path )
			throw_if( 'model', model )
			
			self.input_text = text
			self.file_path = path
			self.instructions = instruct if isinstance( instruct, str ) else ''
			self.model = model
			self.max_tokens = max_tokens
			self.temperature = temperature
			self.include = None
			self.store = store
			self.stream = stream
			self.detail = detail
			self.detail = self.get_analysis_detail( )
			self.client = OpenAI( api_key=self.api_key )
			
			with open( self.file_path, 'rb' ) as source:
				self.file = self.client.files.create( file=source, purpose='vision' )
			
			self.image_content = {
					'type': 'input_image',
					'file_id': self.file.id,
			}
			
			if self.detail:
				self.image_content[ 'detail' ] = self.detail
			
			self.input = [
					{
							'role': 'user',
							'content': [
									{ 'type': 'input_text', 'text': self.input_text },
									self.image_content,
							],
					}
			]
			
			self.request = {
					'model': self.model,
					'input': self.input,
			}
			
			if self.instructions and self.instructions.strip( ):
				self.request[ 'instructions' ] = self.instructions.strip( )
			
			if isinstance( self.max_tokens, int ) and self.max_tokens > 0:
				self.request[ 'max_output_tokens' ] = self.max_tokens
			
			if self.temperature is not None and not self.model.startswith( 'gpt-5' ):
				self.request[ 'temperature' ] = self.temperature
			
			if self.stream is not None:
				self.request[ 'stream' ] = self.stream
			
			if self.store is not None:
				self.request[ 'store' ] = self.store
			
			self.response = self.client.responses.create( **self.request )
			return self.get_output_text( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Images'
			exception.method = 'analyze( self, text: str, path: str, instruct: str=None )'
			Logger( ).write( exception )
			raise exception
	
	def edit( self, prompt: str, path: str, model: str = 'gpt-image-1-mini',
			size: str = '1024x1024', quality: str = 'auto', fmt: str = 'jpeg',
			compression: float = None, background: str = None,
			number: int = None ) -> str | bytes | list[ str | bytes ] | None:
		"""Edit.
		
		Purpose:
			Edits a local source image using the configured OpenAI image model. The method validates
			edit parameters, submits the source image and prompt, and returns the generated image
			output.
		
		Args:
			prompt (str): Prompt value used by the operation.
			path (str): Path value used by the operation.
			model (str): Model value used by the operation.
			size (str): Size value used by the operation.
			quality (str): Quality value used by the operation.
			fmt (str): Fmt value used by the operation.
			compression (float): Compression value used by the operation.
			background (str): Background value used by the operation.
			number (int): Number value used by the operation.
		
		Returns:
			Generated image bytes, URLs, multiple outputs, or no value when the provider returns no
			usable output.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			throw_if( 'prompt', prompt )
			throw_if( 'path', path )
			throw_if( 'model', model )
			throw_if( 'size', size )
			throw_if( 'quality', quality )
			throw_if( 'fmt', fmt )
			
			self.input_text = prompt
			self.file_path = path
			self.model = model
			self.size = size
			self.quality = quality
			self.response_format = fmt
			self.output_format = self.response_format.lower( ).replace( '.', '' )
			self.compression = compression
			self.background = background
			self.number = number
			self.output_compression = self.get_output_compression( )
			
			if not isinstance( self.number, int ) or self.number <= 0:
				self.number = 1
			
			if self.number > 10:
				self.number = 10
			
			if self.model == 'gpt-image-2' and self.background == 'transparent':
				self.background = 'auto'
			
			self.client = OpenAI( api_key=self.api_key )
			self.request = {
					'model': self.model,
					'prompt': self.input_text,
					'size': self.size,
					'quality': self.quality,
					'output_format': self.output_format,
					'n': self.number,
			}
			
			if self.background:
				self.request[ 'background' ] = self.background
			
			if self.output_compression is not None:
				self.request[ 'output_compression' ] = self.output_compression
			
			with open( self.file_path, 'rb' ) as source:
				self.response = self.client.images.edit( image=source, **self.request )
			
			return self.extract_image_outputs( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Images'
			exception.method = 'edit( self, prompt: str, path: str )'
			Logger( ).write( exception )
			raise exception
	
	def __dir__( self ) -> List[ str ] | None:
		"""Return member names.
		
		Purpose:
			Returns a stable list of public members exposed by the Images object for interactive
			inspection, debugging, and application-level compatibility.
		
		Returns:
			Member names exposed for inspection.
		"""
		return [
				'api_key',
				'client',
				'model',
				'prompt',
				'input_text',
				'response',
				'number',
				'size',
				'quality',
				'detail',
				'response_format',
				'mime_format',
				'background',
				'backcolor',
				'compression',
				'image_path',
				'image_url',
				'file',
				'request',
				'output_text',
				'data',
				'outputs',
				'output_format',
				'output_compression',
				'image_content',
				'style_options',
				'model_options',
				'size_options',
				'analysis_model_options',
				'format_options',
				'mime_options',
				'include_options',
				'tool_options',
				'choice_options',
				'backcolor_options',
				'quality_options',
				'detail_options',
				'reasoning_options',
				'modality_options',
				'supports_original_detail',
				'get_analysis_detail',
				'get_output_compression',
				'extract_image_outputs',
				'get_output_text',
				'generate',
				'analyze',
				'edit',
		]

class TTS( GPT ):
	"""Provide TTS workflow support.
	
	Purpose:
		Provides text-to-speech support through the OpenAI Audio Speech API. The class manages
		speech model selection, voice settings, audio format options, temporary streaming
		output, and optional file persistence.
	
	Attributes:
		api_key (Optional[str]): Api key retained by the provider wrapper.
		client (Optional[OpenAI]): Client retained by the provider wrapper.
		speed (Optional[float]): Speed retained by the provider wrapper.
		voice (Optional[str]): Voice retained by the provider wrapper.
		input (Optional[str]): Input retained by the provider wrapper.
		instructions (Optional[str]): Instructions retained by the provider wrapper.
		response (Optional[Any]): Response retained by the provider wrapper.
		response_format (Optional[str]): Response format retained by the provider wrapper.
		file_path (Optional[str]): File path retained by the provider wrapper.
		model (Optional[str]): Model retained by the provider wrapper.
		audio_bytes (Optional[bytes]): Audio bytes retained by the provider wrapper.
		request (Optional[Dict[str, Any]]): Request retained by the provider wrapper.
	"""
	api_key: Optional[ str ]
	client: Optional[ OpenAI ]
	speed: Optional[ float ]
	voice: Optional[ str ]
	input: Optional[ str ]
	instructions: Optional[ str ]
	response: Optional[ Any ]
	response_format: Optional[ str ]
	file_path: Optional[ str ]
	model: Optional[ str ]
	audio_bytes: Optional[ bytes ]
	request: Optional[ Dict[ str, Any ] ]
	
	def __init__( self, input: str = None, model: str = 'gpt-4o-mini-tts', format: str = None,
			instruct: str = None, voice: str = None, speed: float = None, file_path: str = None ):
		"""Initialize instance.
		
		Purpose:
			Initializes the TTS object with default configuration, runtime state, provider settings,
			and compatibility fields. This constructor prepares the instance for later method calls
			without performing external work beyond local attribute assignment.
		
		Args:
			input (str): Input value used by the operation.
			model (str): Model value used by the operation.
			format (str): Format value used by the operation.
			instruct (str): Instruct value used by the operation.
			voice (str): Voice value used by the operation.
			speed (float): Speed value used by the operation.
			file_path (str): File path value used by the operation.
		"""
		super( ).__init__( )
		self.api_key = cfg.OPENAI_API_KEY
		self.client = None
		self.input = input
		self.model = model
		self.instructions = instruct
		self.response_format = format
		self.voice = voice
		self.file_path = file_path
		self.speed = speed
		self.response = None
		self.audio_bytes = None
		self.request = None
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Get model options.
		
		Purpose:
			Returns the model options exposed by the TTS wrapper. The property centralizes UI option
			values and keeps application selectors aligned with the provider-specific
			implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return [
				'gpt-4o-mini-tts',
				'gpt-4o-mini-tts-2025-12-15',
				'tts-1',
				'tts-1-hd',
		]
	
	@property
	def mime_options( self ) -> List[ str ] | None:
		"""Get mime options.
		
		Purpose:
			Returns the mime options exposed by the TTS wrapper. The property centralizes UI option
			values and keeps application selectors aligned with the provider-specific
			implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return [ 'mp3', 'opus', 'aac', 'flac', 'wav', 'pcm', ]
	
	@property
	def voice_options( self ) -> List[ str ] | None:
		"""Get voice options.
		
		Purpose:
			Returns the voice options exposed by the TTS wrapper. The property centralizes UI option
			values and keeps application selectors aligned with the provider-specific
			implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return [
				'alloy',
				'ash',
				'ballad',
				'coral',
				'echo',
				'fable',
				'nova',
				'onyx',
				'sage',
				'shimmer',
				'verse',
				'marin',
				'cedar',
		]
	
	@property
	def speed_options( self ) -> List[ float ] | None:
		"""Get speed options.
		
		Purpose:
			Returns the speed options exposed by the TTS wrapper. The property centralizes UI option
			values and keeps application selectors aligned with the provider-specific
			implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return [
				0.25,
				0.50,
				0.75,
				1.0,
				1.25,
				1.50,
				2.0,
				3.0,
				4.0,
		]
	
	def create_speech( self, text: str, model: str = 'gpt-4o-mini-tts', format: str = 'mp3',
			speed: float = 1.0, voice: str = 'alloy', instruct: str = None,
			file_path: str = None ) -> bytes | None:
		"""Create speech.
		
		Purpose:
			Creates speech audio from text using the configured OpenAI text-to-speech model. The
			method streams the provider response to a temporary file, reads the audio bytes, and
			optionally writes them to a caller-supplied destination.
		
		Args:
			text (str): Text value used by the operation.
			model (str): Model value used by the operation.
			format (str): Format value used by the operation.
			speed (float): Speed value used by the operation.
			voice (str): Voice value used by the operation.
			instruct (str): Instruct value used by the operation.
			file_path (str): File path value used by the operation.
		
		Returns:
			Generated audio bytes when available.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			throw_if( 'text', text )
			throw_if( 'model', model )
			throw_if( 'format', format )
			throw_if( 'voice', voice )
			
			self.input = text
			self.model = model
			self.response_format = format
			self.voice = voice
			self.speed = speed
			self.instructions = instruct
			self.file_path = file_path
			self.client = OpenAI( api_key=self.api_key )
			self.response = None
			self.audio_bytes = None
			
			with tempfile.NamedTemporaryFile(
					suffix=f'.{self.response_format}', delete=False ) as tmp:
				temp_path = tmp.name
			
			try:
				self.request = {
						'model': self.model,
						'voice': self.voice,
						'input': self.input,
						'response_format': self.response_format,
						'speed': self.speed,
				}
				
				if self.instructions and self.model not in ('tts-1', 'tts-1-hd'):
					self.request[ 'instructions' ] = self.instructions
				
				with self.client.audio.speech.with_streaming_response.create(
						**self.request ) as response:
					self.response = response
					response.stream_to_file( temp_path )
				
				with open( temp_path, 'rb' ) as source:
					self.audio_bytes = source.read( )
				
				if self.file_path:
					with open( self.file_path, 'wb' ) as target:
						target.write( self.audio_bytes )
				
				return self.audio_bytes
			finally:
				try:
					if os.path.exists( temp_path ):
						os.remove( temp_path )
				except Exception as e:
					exception = Error( e )
					exception.module = 'gpt'
					exception.cause = 'TTS'
					exception.method = 'create_speech( ... )'
					Logger( ).write( exception )
					pass
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'TTS'
			exception.method = 'create_speech( self, text: str ) -> bytes | None'
			Logger( ).write( exception )
			raise exception
	
	def __dir__( self ) -> List[ str ] | None:
		"""Return member names.
		
		Purpose:
			Returns a stable list of public members exposed by the TTS object for interactive
			inspection, debugging, and application-level compatibility.
		
		Returns:
			Member names exposed for inspection.
		"""
		return [
				'input',
				'file_path',
				'voice',
				'client',
				'response_format',
				'speed',
				'model',
				'instructions',
				'response',
				'audio_bytes',
				'request',
				'model_options',
				'mime_options',
				'voice_options',
				'speed_options',
				'create_speech',
		]

class Transcription( GPT ):
	"""Provide Transcription workflow support.
	
	Purpose:
		Provides audio transcription support through the OpenAI Audio Transcriptions API. The
		class manages transcription model options, source-language hints, response formats,
		include fields, and normalized transcript output.
	
	Attributes:
		client (Optional[OpenAI]): Client retained by the provider wrapper.
		language (Optional[str]): Language retained by the provider wrapper.
		instructions (Optional[str]): Instructions retained by the provider wrapper.
		include (Optional[List[str]]): Include retained by the provider wrapper.
		normalized_result (Optional[Dict[str, Any]]): Normalized result retained by the provider wrapper.
	"""
	client: Optional[ OpenAI ]
	language: Optional[ str ]
	instructions: Optional[ str ]
	include: Optional[ List[ str ] ]
	normalized_result: Optional[ Dict[ str, Any ] ]
	
	def __init__( self, model: str = 'gpt-4o-transcribe', temperature: float = None,
			prompt: str = None, number: int = None, top_p: float = None, frequency: float = None,
			presence: float = None, max_tokens: int = None, stream: bool = None, store: bool = None,
			language: str = None, instruct: str = None, format: str = None, background: bool = None,
			messages: List[ Dict[ str, str ] ] = None, stops: List[ str ] = None,
			include: List[ str ] = None ):
		"""Initialize instance.
		
		Purpose:
			Initializes the Transcription object with default configuration, runtime state, provider
			settings, and compatibility fields. This constructor prepares the instance for later
			method calls without performing external work beyond local attribute assignment.
		
		Args:
			model (str): Model value used by the operation.
			temperature (float): Temperature value used by the operation.
			prompt (str): Prompt value used by the operation.
			number (int): Number value used by the operation.
			top_p (float): Top p value used by the operation.
			frequency (float): Frequency value used by the operation.
			presence (float): Presence value used by the operation.
			max_tokens (int): Max tokens value used by the operation.
			stream (bool): Stream value used by the operation.
			store (bool): Store value used by the operation.
			language (str): Language value used by the operation.
			instruct (str): Instruct value used by the operation.
			format (str): Format value used by the operation.
			background (bool): Background value used by the operation.
			messages (List[Dict[str, str]]): Messages value used by the operation.
			stops (List[str]): Stops value used by the operation.
			include (List[str]): Include value used by the operation.
		"""
		super( ).__init__( )
		self.api_key = cfg.OPENAI_API_KEY
		self.prompt = prompt
		self.temperature = temperature
		self.top_percent = top_p
		self.frequency_penalty = frequency
		self.presence_penalty = presence
		self.max_tokens = max_tokens
		self.stream = stream
		self.response_format = format
		self.background = background
		self.message = messages
		self.stops = stops
		self.store = store
		self.language = language
		self.instructions = instruct
		self.model = model
		self.number = number
		self.input_text = None
		self.audio_file = None
		self.transcript = None
		self.response = None
		self.include = include if include is not None else [ ]
		self.normalized_result = None
		self.request = None
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Get model options.
		
		Purpose:
			Returns the model options exposed by the Transcription wrapper. The property centralizes
			UI option values and keeps application selectors aligned with the provider-specific
			implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return [
				'gpt-4o-transcribe',
				'gpt-4o-mini-transcribe',
				'gpt-4o-mini-transcribe-2025-12-15',
				'whisper-1',
				'gpt-4o-transcribe-diarize',
		]
	
	@property
	def mime_options( self ) -> List[ str ] | None:
		"""Get mime options.
		
		Purpose:
			Returns the mime options exposed by the Transcription wrapper. The property centralizes
			UI option values and keeps application selectors aligned with the provider-specific
			implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return [
				'flac',
				'mp3',
				'mp4',
				'mpeg',
				'mpga',
				'm4a',
				'ogg',
				'wav',
				'webm',
		]
	
	@property
	def language_options( self ) -> List[ str ] | None:
		"""Get language options.
		
		Purpose:
			Returns the language options exposed by the Transcription wrapper. The property
			centralizes UI option values and keeps application selectors aligned with the
			provider-specific implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return [
				'en',
				'es',
				'fr',
				'de',
				'it',
				'pt',
				'ru',
				'uk',
				'el',
				'he',
				'ar',
				'hi',
				'zh',
				'ja',
				'ko',
				'vi',
				'th',
		]
	
	@property
	def language_labels( self ) -> Dict[ str, str ] | None:
		"""Get language labels.
		
		Purpose:
			Returns the language labels exposed by the Transcription wrapper. The property
			centralizes UI option values and keeps application selectors aligned with the
			provider-specific implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return {
				'en': 'English',
				'es': 'Spanish',
				'fr': 'French',
				'de': 'German',
				'it': 'Italian',
				'pt': 'Portuguese',
				'ru': 'Russian',
				'uk': 'Ukrainian',
				'el': 'Greek',
				'he': 'Hebrew',
				'ar': 'Arabic',
				'hi': 'Hindi',
				'zh': 'Chinese',
				'ja': 'Japanese',
				'ko': 'Korean',
				'vi': 'Vietnamese',
				'th': 'Thai',
		}
	
	@property
	def include_options( self ) -> List[ str ] | None:
		"""Get include options.
		
		Purpose:
			Returns the include options exposed by the Transcription wrapper. The property
			centralizes UI option values and keeps application selectors aligned with the
			provider-specific implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return [
				'logprobs',
		]
	
	@property
	def response_format_options( self ) -> Dict[ str, List[ str ] ]:
		"""Get response format options.
		
		Purpose:
			Returns the response format options exposed by the Transcription wrapper. The property
			centralizes UI option values and keeps application selectors aligned with the
			provider-specific implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return {
				'whisper-1': [
						'json',
						'text',
						'srt',
						'verbose_json',
						'vtt',
				],
				'gpt-4o-transcribe': [
						'json',
				],
				'gpt-4o-mini-transcribe': [
						'json',
				],
				'gpt-4o-mini-transcribe-2025-12-15': [
						'json',
				],
				'gpt-4o-transcribe-diarize': [
						'json',
						'text',
						'diarized_json',
				],
		}
	
	def get_include( self ) -> List[ str ]:
		"""Get include.
		
		Purpose:
			Returns the include value for the active Transcription request. The method inspects
			current runtime state and provides a safe application-facing result.
		
		Returns:
			Requested value derived from the current runtime state.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			if self.include is None or len( self.include ) == 0:
				return [ ]
			
			if self.model not in [
					'gpt-4o-transcribe',
					'gpt-4o-mini-transcribe',
					'gpt-4o-mini-transcribe-2025-12-15',
			]:
				return [ ]
			
			values = [ ]
			for item in self.include:
				if isinstance( item, str ) and item.strip( ) == 'logprobs':
					values.append( item.strip( ) )
			
			return values
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Transcription'
			exception.method = 'get_include( self ) -> List[ str ]'
			Logger( ).write( exception )
			raise exception
	
	def normalize_response( self, response: Any ) -> Dict[ str, Any ]:
		"""Normalize response.
		
		Purpose:
			Normalizes the response value used for the Transcription workflow. The method converts
			provider-specific objects, dictionaries, or compatibility inputs into a stable structure
			for application use.
		
		Args:
			response (Any): Response value used by the operation.
		
		Returns:
			Normalized application-facing value or structure.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			result: Dict[ str, Any ] = {
					'text': '',
					'segments': [ ],
					'language': None,
					'duration': None,
					'raw': None,
			}
			
			if response is None:
				return result
			
			if isinstance( response, str ):
				result[ 'text' ] = response
				result[ 'raw' ] = response
				return result
			
			if hasattr( response, 'model_dump' ):
				try:
					result[ 'raw' ] = response.model_dump( )
				except Exception as e:
					exception = Error( e )
					exception.module = 'gpt'
					exception.cause = 'Transcription'
					exception.method = 'normalize_response( ... )'
					Logger( ).write( exception )
					result[ 'raw' ] = str( response )
			else:
				result[ 'raw' ] = str( response )
			
			text = getattr( response, 'text', None )
			if isinstance( text, str ):
				result[ 'text' ] = text
			
			segments = getattr( response, 'segments', None )
			if isinstance( segments, list ):
				normalized_segments = [ ]
				for segment in segments:
					if hasattr( segment, 'model_dump' ):
						normalized_segments.append( segment.model_dump( ) )
					elif isinstance( segment, dict ):
						normalized_segments.append( segment )
					else:
						normalized_segments.append( { 'text': str( segment ) } )
				
				result[ 'segments' ] = normalized_segments
			
			language = getattr( response, 'language', None )
			if language:
				result[ 'language' ] = language
			
			duration = getattr( response, 'duration', None )
			if duration:
				result[ 'duration' ] = duration
			
			if not result[ 'text' ] and len( result[ 'segments' ] ) > 0:
				parts = [ ]
				for segment in result[ 'segments' ]:
					if isinstance( segment, dict ) and segment.get( 'text' ):
						parts.append( str( segment.get( 'text' ) ) )
				
				result[ 'text' ] = '\n'.join( parts ).strip( )
			
			if not result[ 'text' ]:
				result[ 'text' ] = str( response )
			
			return result
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Transcription'
			exception.method = 'normalize_response( self, response: Any ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def transcribe( self, path: str, model: str = 'gpt-4o-transcribe', language: str = None,
			prompt: str = None, format: str = None, temperature: float = None,
			include: List[ str ] = None ) -> str | None:
		"""Transcribe.
		
		Purpose:
			Transcribes a local audio file using the configured OpenAI transcription model. The
			method builds the transcription request, normalizes the provider response, and returns
			the extracted transcript text.
		
		Args:
			path (str): Path value used by the operation.
			model (str): Model value used by the operation.
			language (str): Language value used by the operation.
			prompt (str): Prompt value used by the operation.
			format (str): Format value used by the operation.
			temperature (float): Temperature value used by the operation.
			include (List[str]): Include value used by the operation.
		
		Returns:
			Generated or extracted text when available.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			throw_if( 'path', path )
			throw_if( 'model', model )
			
			self.audio_file = path
			self.model = model
			self.language = language if isinstance( language, str ) and language.strip( ) else None
			self.prompt = prompt if isinstance( prompt, str ) and prompt.strip( ) else None
			self.response_format = format if isinstance( format,
				str ) and format.strip( ) else 'json'
			self.temperature = temperature
			self.include = include if include is not None else [ ]
			self.include = self.get_include( )
			self.client = OpenAI( api_key=cfg.OPENAI_API_KEY )
			self.request = { 'model': self.model, }
			
			if self.language:
				self.request[ 'language' ] = self.language
			
			if self.prompt:
				self.request[ 'prompt' ] = self.prompt
			
			if self.response_format:
				self.request[ 'response_format' ] = self.response_format
			
			if self.include:
				self.request[ 'include' ] = self.include
			
			if self.temperature is not None and self.model == 'whisper-1':
				self.request[ 'temperature' ] = self.temperature
			
			with open( self.audio_file, 'rb' ) as source:
				self.response = self.client.audio.transcriptions.create(
					file=source,
					**self.request )
			
			self.normalized_result = self.normalize_response( self.response )
			self.transcript = self.normalized_result.get( 'text' )
			return self.transcript
		except Exception as e:
			ex = Error( e )
			ex.module = 'gpt'
			ex.cause = 'Transcription'
			ex.method = 'transcribe( self, path: str ) -> str | None'
			Logger( ).write( ex )
			raise ex
	
	def __dir__( self ) -> List[ str ] | None:
		"""Return member names.
		
		Purpose:
			Returns a stable list of public members exposed by the Transcription object for
			interactive inspection, debugging, and application-level compatibility.
		
		Returns:
			Member names exposed for inspection.
		"""
		return [
				'number',
				'temperature',
				'top_percent',
				'frequency_penalty',
				'presence_penalty',
				'max_tokens',
				'store',
				'stream',
				'stops',
				'prompt',
				'response',
				'audio_file',
				'messages',
				'response_format',
				'api_key',
				'client',
				'input_text',
				'transcript',
				'language',
				'model',
				'include',
				'normalized_result',
				'model_options',
				'mime_options',
				'language_options',
				'language_labels',
				'include_options',
				'response_format_options',
				'get_include',
				'normalize_response',
				'transcribe',
		]

class Translation( GPT ):
	"""Provide Translation workflow support.
	
	Purpose:
		Provides audio translation support through the OpenAI Audio Translations API. The class
		manages translation model settings, response formatting, source-language context, and
		normalized English output.
	
	Attributes:
		client (Optional[OpenAI]): Client retained by the provider wrapper.
		target_language (Optional[str]): Target language retained by the provider wrapper.
		response_format (Optional[str]): Response format retained by the provider wrapper.
		normalized_result (Optional[Dict[str, Any]]): Normalized result retained by the provider wrapper.
	"""
	client: Optional[ OpenAI ]
	target_language: Optional[ str ]
	response_format: Optional[ str ]
	normalized_result: Optional[ Dict[ str, Any ] ]
	
	def __init__( self, model: str = 'whisper-1', temperature: float = None, top_p: float = None,
			frequency: float = None, presence: float = None, max_tokens: int = None,
			store: bool = None, stream: bool = None, instruct: str = None, audio_file: str = None,
			format: str = None, language: str = None ):
		"""Initialize instance.
		
		Purpose:
			Initializes the Translation object with default configuration, runtime state, provider
			settings, and compatibility fields. This constructor prepares the instance for later
			method calls without performing external work beyond local attribute assignment.
		
		Args:
			model (str): Model value used by the operation.
			temperature (float): Temperature value used by the operation.
			top_p (float): Top p value used by the operation.
			frequency (float): Frequency value used by the operation.
			presence (float): Presence value used by the operation.
			max_tokens (int): Max tokens value used by the operation.
			store (bool): Store value used by the operation.
			stream (bool): Stream value used by the operation.
			instruct (str): Instruct value used by the operation.
			audio_file (str): Audio file value used by the operation.
			format (str): Format value used by the operation.
			language (str): Language value used by the operation.
		"""
		super( ).__init__( )
		self.api_key = cfg.OPENAI_API_KEY
		self.client = None
		self.model = model
		self.temperature = temperature
		self.top_percent = top_p
		self.frequency_penalty = frequency
		self.presence_penalty = presence
		self.max_tokens = max_tokens
		self.store = store
		self.stream = stream
		self.instructions = instruct
		self.audio_file = audio_file
		self.response = None
		self.response_format = format
		self.target_language = language
		self.normalized_result = None
		self.request = None
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Get model options.
		
		Purpose:
			Returns the model options exposed by the Translation wrapper. The property centralizes
			UI option values and keeps application selectors aligned with the provider-specific
			implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return [ 'whisper-1', ]
	
	@property
	def mime_options( self ) -> List[ str ] | None:
		"""Get mime options.
		
		Purpose:
			Returns the mime options exposed by the Translation wrapper. The property centralizes UI
			option values and keeps application selectors aligned with the provider-specific
			implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return [
				'flac',
				'mp3',
				'mp4',
				'mpeg',
				'mpga',
				'm4a',
				'ogg',
				'wav',
				'webm',
		]
	
	@property
	def language_options( self ) -> List[ str ] | None:
		"""Get language options.
		
		Purpose:
			Returns the language options exposed by the Translation wrapper. The property
			centralizes UI option values and keeps application selectors aligned with the
			provider-specific implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return [
				'en',
				'es',
				'fr',
				'de',
				'it',
				'pt',
				'ru',
				'uk',
				'el',
				'he',
				'ar',
				'hi',
				'zh',
				'ja',
				'ko',
				'vi',
				'th',
		]
	
	@property
	def language_labels( self ) -> Dict[ str, str ] | None:
		"""Get language labels.
		
		Purpose:
			Returns the language labels exposed by the Translation wrapper. The property centralizes
			UI option values and keeps application selectors aligned with the provider-specific
			implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return {
				'en': 'English',
				'es': 'Spanish',
				'fr': 'French',
				'de': 'German',
				'it': 'Italian',
				'pt': 'Portuguese',
				'ru': 'Russian',
				'uk': 'Ukrainian',
				'el': 'Greek',
				'he': 'Hebrew',
				'ar': 'Arabic',
				'hi': 'Hindi',
				'zh': 'Chinese',
				'ja': 'Japanese',
				'ko': 'Korean',
				'vi': 'Vietnamese',
				'th': 'Thai',
		}
	
	@property
	def response_format_options( self ) -> List[ str ] | None:
		"""Get response format options.
		
		Purpose:
			Returns the response format options exposed by the Translation wrapper. The property
			centralizes UI option values and keeps application selectors aligned with the
			provider-specific implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return [
				'json',
				'text',
				'srt',
				'verbose_json',
				'vtt',
		]
	
	def normalize_response( self, response: Any ) -> Dict[ str, Any ]:
		"""Normalize response.
		
		Purpose:
			Normalizes the response value used for the Translation workflow. The method converts
			provider-specific objects, dictionaries, or compatibility inputs into a stable structure
			for application use.
		
		Args:
			response (Any): Response value used by the operation.
		
		Returns:
			Normalized application-facing value or structure.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			result: Dict[ str, Any ] = {
					'text': '',
					'segments': [ ],
					'language': None,
					'duration': None,
					'raw': None,
			}
			
			if response is None:
				return result
			
			if isinstance( response, str ):
				result[ 'text' ] = response
				result[ 'raw' ] = response
				return result
			
			if hasattr( response, 'model_dump' ):
				try:
					result[ 'raw' ] = response.model_dump( )
				except Exception as e:
					exception = Error( e )
					exception.module = 'gpt'
					exception.cause = 'Translation'
					exception.method = 'normalize_response( ... )'
					Logger( ).write( exception )
					result[ 'raw' ] = str( response )
			else:
				result[ 'raw' ] = str( response )
			
			text = getattr( response, 'text', None )
			if isinstance( text, str ):
				result[ 'text' ] = text
			
			segments = getattr( response, 'segments', None )
			if isinstance( segments, list ):
				normalized_segments = [ ]
				for segment in segments:
					if hasattr( segment, 'model_dump' ):
						normalized_segments.append( segment.model_dump( ) )
					elif isinstance( segment, dict ):
						normalized_segments.append( segment )
					else:
						normalized_segments.append( { 'text': str( segment ) } )
				
				result[ 'segments' ] = normalized_segments
			
			language = getattr( response, 'language', None )
			if language:
				result[ 'language' ] = language
			
			duration = getattr( response, 'duration', None )
			if duration:
				result[ 'duration' ] = duration
			
			if not result[ 'text' ] and len( result[ 'segments' ] ) > 0:
				parts = [ ]
				for segment in result[ 'segments' ]:
					if isinstance( segment, dict ) and segment.get( 'text' ):
						parts.append( str( segment.get( 'text' ) ) )
				
				result[ 'text' ] = '\n'.join( parts ).strip( )
			
			if not result[ 'text' ]:
				result[ 'text' ] = str( response )
			
			return result
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Translation'
			exception.method = 'normalize_response( self, response: Any ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def translate( self, filepath: str, model: str = 'whisper-1', prompt: str = None,
			format: str = None, temperature: float = None, language: str = None ) -> str | None:
		"""Translate.
		
		Purpose:
			Translates non-English speech from a local audio file into English using the OpenAI
			translation API. The method normalizes the provider response and returns the translated
			text when available.
		
		Args:
			filepath (str): Filepath value used by the operation.
			model (str): Model value used by the operation.
			prompt (str): Prompt value used by the operation.
			format (str): Format value used by the operation.
			temperature (float): Temperature value used by the operation.
			language (str): Language value used by the operation.
		
		Returns:
			Generated or extracted text when available.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			throw_if( 'filepath', filepath )
			throw_if( 'model', model )
			
			self.audio_file = filepath
			self.model = model
			self.prompt = prompt if isinstance( prompt, str ) and prompt.strip( ) else None
			self.response_format = format if isinstance( format,
				str ) and format.strip( ) else 'json'
			self.temperature = temperature
			self.target_language = language
			self.client = OpenAI( api_key=self.api_key )
			self.request = { 'model': self.model, }
			
			if self.prompt:
				self.request[ 'prompt' ] = self.prompt
			
			if self.response_format:
				self.request[ 'response_format' ] = self.response_format
			
			if self.temperature is not None:
				self.request[ 'temperature' ] = self.temperature
			
			with open( self.audio_file, 'rb' ) as source:
				self.response = self.client.audio.translations.create( file=source, **self.request )
			
			self.normalized_result = self.normalize_response( self.response )
			return self.normalized_result.get( 'text' )
		except Exception as e:
			ex = Error( e )
			ex.module = 'gpt'
			ex.cause = 'Translation'
			ex.method = 'translate( self, filepath: str ) -> str | None'
			Logger( ).write( ex )
			raise ex
	
	def __dir__( self ) -> List[ str ] | None:
		"""Return member names.
		
		Purpose:
			Returns a stable list of public members exposed by the Translation object for
			interactive inspection, debugging, and application-level compatibility.
		
		Returns:
			Member names exposed for inspection.
		"""
		return [
				'temperature',
				'top_percent',
				'frequency_penalty',
				'presence_penalty',
				'max_tokens',
				'store',
				'stream',
				'prompt',
				'response',
				'audio_file',
				'response_format',
				'api_key',
				'client',
				'model',
				'target_language',
				'normalized_result',
				'model_options',
				'mime_options',
				'language_options',
				'language_labels',
				'response_format_options',
				'normalize_response',
				'translate',
		]

class Embeddings( GPT ):
	"""Provide Embeddings workflow support.
	
	Purpose:
		Provides OpenAI embedding generation for text inputs. The class manages embedding model
		selection, encoding format, optional dimensions, usage metadata, and normalized single
		or batch embedding output.
	
	Attributes:
		api_key (Optional[str]): Api key retained by the provider wrapper.
		client (Optional[OpenAI]): Client retained by the provider wrapper.
		model (Optional[str]): Model retained by the provider wrapper.
		input (Optional[str | List[str]]): Input retained by the provider wrapper.
		encoding_format (Optional[str]): Encoding format retained by the provider wrapper.
		dimensions (Optional[int]): Dimensions retained by the provider wrapper.
		user (Optional[str]): User retained by the provider wrapper.
		response (Optional[CreateEmbeddingResponse]): Response retained by the provider wrapper.
		embedding (Optional[List[float] | str]): Embedding retained by the provider wrapper.
		embeddings (Optional[List[List[float]] | List[str]]): Embeddings retained by the provider wrapper.
		usage (Optional[Any]): Usage retained by the provider wrapper.
		request (Optional[Dict[str, Any]]): Request retained by the provider wrapper.
	"""
	api_key: Optional[ str ]
	client: Optional[ OpenAI ]
	model: Optional[ str ]
	input: Optional[ str | List[ str ] ]
	encoding_format: Optional[ str ]
	dimensions: Optional[ int ]
	user: Optional[ str ]
	response: Optional[ CreateEmbeddingResponse ]
	embedding: Optional[ List[ float ] | str ]
	embeddings: Optional[ List[ List[ float ] ] | List[ str ] ]
	usage: Optional[ Any ]
	request: Optional[ Dict[ str, Any ] ]
	
	def __init__( self, text: str | List[ str ] = None, model: str = 'text-embedding-3-small',
			format: str = 'float', dimensions: int = None, user: str = None ):
		"""Initialize instance.
		
		Purpose:
			Initializes the Embeddings object with default configuration, runtime state, provider
			settings, and compatibility fields. This constructor prepares the instance for later
			method calls without performing external work beyond local attribute assignment.
		
		Args:
			text (str | List[str]): Text value used by the operation.
			model (str): Model value used by the operation.
			format (str): Format value used by the operation.
			dimensions (int): Dimensions value used by the operation.
			user (str): User value used by the operation.
		"""
		super( ).__init__( )
		self.api_key = cfg.OPENAI_API_KEY
		self.client = None
		self.model = model
		self.input = text
		self.encoding_format = format
		self.dimensions = dimensions
		self.user = user
		self.response = None
		self.embedding = None
		self.embeddings = None
		self.usage = None
		self.request = None
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Get model options.
		
		Purpose:
			Returns the model options exposed by the Embeddings wrapper. The property centralizes UI
			option values and keeps application selectors aligned with the provider-specific
			implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return [ 'text-embedding-3-small', 'text-embedding-3-large', 'text-embedding-ada-002', ]
	
	@property
	def encoding_options( self ) -> List[ str ] | None:
		"""Get encoding options.
		
		Purpose:
			Returns the encoding options exposed by the Embeddings wrapper. The property centralizes
			UI option values and keeps application selectors aligned with the provider-specific
			implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return [ 'float', 'base64', ]
	
	@property
	def model_default_dimensions( self ) -> Dict[ str, int ]:
		"""Get model default dimensions.
		
		Purpose:
			Returns the model default dimensions exposed by the Embeddings wrapper. The property
			centralizes UI option values and keeps application selectors aligned with the
			provider-specific implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return {
				'text-embedding-3-small': 1536,
				'text-embedding-3-large': 3072,
				'text-embedding-ada-002': 1536,
		}
	
	@property
	def model_max_dimensions( self ) -> Dict[ str, int ]:
		"""Get model max dimensions.
		
		Purpose:
			Returns the model max dimensions exposed by the Embeddings wrapper. The property
			centralizes UI option values and keeps application selectors aligned with the
			provider-specific implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return {
				'text-embedding-3-small': 1536,
				'text-embedding-3-large': 3072,
				'text-embedding-ada-002': 1536,
		}
	
	@property
	def model_dimension_support( self ) -> Dict[ str, bool ]:
		"""Get model dimension support.
		
		Purpose:
			Returns the model dimension support exposed by the Embeddings wrapper. The property
			centralizes UI option values and keeps application selectors aligned with the
			provider-specific implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return {
				'text-embedding-3-small': True,
				'text-embedding-3-large': True,
				'text-embedding-ada-002': False,
		}
	
	def validate_input( self, text: str | List[ str ] ) -> str | List[ str ]:
		"""Validate input.
		
		Purpose:
			Validates and normalizes the input value used for the Embeddings workflow. The method
			raises an application error when required input is missing and returns a clean value
			suitable for downstream provider calls.
		
		Args:
			text (str | List[str]): Text value used by the operation.
		
		Returns:
			Validated and normalized value for downstream use.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			throw_if( 'text', text )
			
			if isinstance( text, str ):
				value = text.strip( )
				throw_if( 'text', value )
				return value
			
			if isinstance( text, list ):
				values = [ ]
				for item in text:
					if not isinstance( item, str ):
						continue
					
					clean = item.strip( )
					if clean:
						values.append( clean )
				
				throw_if( 'text', values )
				return values
			
			raise ValueError( 'Embedding input must be a string or list of strings.' )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Embeddings'
			exception.method = 'validate_input( self, text: str | List[ str ] )'
			Logger( ).write( exception )
			raise exception
	
	def validate_dimensions( self ) -> int | None:
		"""Validate dimensions.
		
		Purpose:
			Validates and normalizes the dimensions value used for the Embeddings workflow. The
			method raises an application error when required input is missing and returns a clean
			value suitable for downstream provider calls.
		
		Returns:
			Validated and normalized value for downstream use.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			if self.dimensions is None:
				return None
			
			try:
				value = int( self.dimensions )
			except Exception as e:
				exception = Error( e )
				exception.module = 'gpt'
				exception.cause = 'Embeddings'
				exception.method = 'validate_dimensions( ... )'
				Logger( ).write( exception )
				return None
			
			if value <= 0:
				return None
			
			supports_dimensions = self.model_dimension_support.get( self.model, False )
			if not supports_dimensions:
				return None
			
			max_dimensions = self.get_max_dimensions( self.model )
			if value > max_dimensions:
				return max_dimensions
			
			return value
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Embeddings'
			exception.method = 'validate_dimensions( self ) -> int | None'
			Logger( ).write( exception )
			raise exception
	
	def get_default_dimensions( self, model: str ) -> int:
		"""Get default dimensions.
		
		Purpose:
			Returns the default dimensions value for the active Embeddings request. The method
			inspects current runtime state and provides a safe application-facing result.
		
		Args:
			model (str): Model value used by the operation.
		
		Returns:
			Requested value derived from the current runtime state.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			throw_if( 'model', model )
			return int( self.model_default_dimensions.get( model, 1536 ) )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Embeddings'
			exception.method = 'get_default_dimensions( self, model: str ) -> int'
			Logger( ).write( exception )
			raise exception
	
	def get_max_dimensions( self, model: str ) -> int:
		"""Get max dimensions.
		
		Purpose:
			Returns the max dimensions value for the active Embeddings request. The method inspects
			current runtime state and provides a safe application-facing result.
		
		Args:
			model (str): Model value used by the operation.
		
		Returns:
			Requested value derived from the current runtime state.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			throw_if( 'model', model )
			return int( self.model_max_dimensions.get( model, 1536 ) )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Embeddings'
			exception.method = 'get_max_dimensions( self, model: str ) -> int'
			Logger( ).write( exception )
			raise exception
	
	def build_request( self, text: str | List[ str ], model: str = 'text-embedding-3-small',
			format: str = 'float', dimensions: int = None, user: str = None ) -> Dict[ str, Any ]:
		"""Build request.
		
		Purpose:
			Builds the request payload used for the Embeddings workflow. The method validates caller
			input, applies compatibility defaults, and returns a provider-ready structure without
			executing the provider request.
		
		Args:
			text (str | List[str]): Text value used by the operation.
			model (str): Model value used by the operation.
			format (str): Format value used by the operation.
			dimensions (int): Dimensions value used by the operation.
			user (str): User value used by the operation.
		
		Returns:
			Provider-ready request structure or omitted optional payload.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			throw_if( 'text', text )
			throw_if( 'model', model )
			throw_if( 'format', format )
			
			self.input = self.validate_input( text )
			self.model = model
			self.encoding_format = format
			self.dimensions = dimensions
			self.dimensions = self.validate_dimensions( )
			self.user = user if isinstance( user, str ) and user.strip( ) else None
			self.request = {
					'model': self.model,
					'input': self.input,
					'encoding_format': self.encoding_format,
			}
			
			if self.dimensions is not None:
				self.request[ 'dimensions' ] = self.dimensions
			
			if self.user:
				self.request[ 'user' ] = self.user.strip( )
			
			return self.request
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Embeddings'
			exception.method = 'build_request( self, text: str | List[ str ], **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def create( self, text: str | List[ str ], model: str = 'text-embedding-3-small',
			format: str = 'float', dimensions: int = None,
			user: str = None ) -> List[ float ] | List[ List[ float ] ] | str | List[ str ] | None:
		"""Create.
		
		Purpose:
			Creates provider resources or generated outputs for the Embeddings workflow using
			validated request state and provider-specific defaults.
		
		Args:
			text (str | List[str]): Text value used by the operation.
			model (str): Model value used by the operation.
			format (str): Format value used by the operation.
			dimensions (int): Dimensions value used by the operation.
			user (str): User value used by the operation.
		
		Returns:
			Single embedding, batch embeddings, base64 embedding content, or no value when no
			embeddings are returned.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			self.client = OpenAI( api_key=self.api_key )
			self.request = self.build_request( text=text, model=model, format=format,
				dimensions=dimensions, user=user )
			
			self.response = self.client.embeddings.create( **self.request )
			self.usage = getattr( self.response, 'usage', None )
			self.data = getattr( self.response, 'data', None )
			self.embeddings = [ ]
			
			if self.data is None or len( self.data ) == 0:
				self.embedding = None
				return None
			
			for item in self.data:
				value = getattr( item, 'embedding', None )
				if value is not None:
					self.embeddings.append( value )
			
			if len( self.embeddings ) == 0:
				self.embedding = None
				return None
			
			self.embedding = self.embeddings[ 0 ]
			
			if isinstance( self.input, str ):
				return self.embedding
			
			return self.embeddings
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Embeddings'
			exception.method = 'create( self, text: str | List[ str ], **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def __dir__( self ) -> List[ str ] | None:
		"""Return member names.
		
		Purpose:
			Returns a stable list of public members exposed by the Embeddings object for interactive
			inspection, debugging, and application-level compatibility.
		
		Returns:
			Member names exposed for inspection.
		"""
		return [
				'api_key',
				'client',
				'model',
				'input',
				'encoding_format',
				'dimensions',
				'user',
				'response',
				'embedding',
				'embeddings',
				'usage',
				'request',
				'model_options',
				'encoding_options',
				'model_default_dimensions',
				'model_max_dimensions',
				'model_dimension_support',
				'validate_input',
				'validate_dimensions',
				'get_default_dimensions',
				'get_max_dimensions',
				'build_request',
				'create',
		]

class Files( GPT ):
	"""Provide Files workflow support.
	
	Purpose:
		Provides OpenAI Files API support for upload, listing, retrieval, extraction, deletion,
		and file-content analysis workflows. The class stores selected file metadata, content
		previews, and Responses API analysis output.
	
	Attributes:
		api_key (Optional[str]): Api key retained by the provider wrapper.
		client (Optional[OpenAI]): Client retained by the provider wrapper.
		file (Optional[Any]): File retained by the provider wrapper.
		file_id (Optional[str]): File id retained by the provider wrapper.
		filepath (Optional[str]): Filepath retained by the provider wrapper.
		filename (Optional[str]): Filename retained by the provider wrapper.
		purpose (Optional[str]): Purpose retained by the provider wrapper.
		response (Optional[Any]): Response retained by the provider wrapper.
		content (Optional[str | bytes | Dict[str, Any]]): Content retained by the provider wrapper.
		files (Optional[List[Dict[str, Any]]]): Files retained by the provider wrapper.
		request (Optional[Dict[str, Any]]): Request retained by the provider wrapper.
		model (Optional[str]): Model retained by the provider wrapper.
		prompt (Optional[str]): Prompt retained by the provider wrapper.
		output_text (Optional[str]): Output text retained by the provider wrapper.
	"""
	api_key: Optional[ str ]
	client: Optional[ OpenAI ]
	file: Optional[ Any ]
	file_id: Optional[ str ]
	filepath: Optional[ str ]
	filename: Optional[ str ]
	purpose: Optional[ str ]
	response: Optional[ Any ]
	content: Optional[ str | bytes | Dict[ str, Any ] ]
	files: Optional[ List[ Dict[ str, Any ] ] ]
	request: Optional[ Dict[ str, Any ] ]
	model: Optional[ str ]
	prompt: Optional[ str ]
	output_text: Optional[ str ]
	
	def __init__( self, id: str = None, filepath: str = None, purpose: str = 'user_data',
			model: str = 'gpt-4o-mini', prompt: str = None ):
		"""Initialize instance.
		
		Purpose:
			Initializes the Files object with default configuration, runtime state, provider
			settings, and compatibility fields. This constructor prepares the instance for later
			method calls without performing external work beyond local attribute assignment.
		
		Args:
			id (str): Id value used by the operation.
			filepath (str): Filepath value used by the operation.
			purpose (str): Purpose value used by the operation.
			model (str): Model value used by the operation.
			prompt (str): Prompt value used by the operation.
		"""
		super( ).__init__( )
		self.api_key = cfg.OPENAI_API_KEY
		self.client = None
		self.file = None
		self.file_id = id
		self.filepath = filepath
		self.filename = None
		self.purpose = purpose
		self.response = None
		self.content = None
		self.files = [ ]
		self.request = None
		self.model = model
		self.prompt = prompt
		self.output_text = None
	
	@property
	def upload_purpose_options( self ) -> List[ str ] | None:
		"""Get upload purpose options.
		
		Purpose:
			Returns the upload purpose options exposed by the Files wrapper. The property
			centralizes UI option values and keeps application selectors aligned with the
			provider-specific implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return [
				'assistants',
				'batch',
				'fine-tune',
				'vision',
				'user_data',
				'evals',
		]
	
	@property
	def file_purpose_options( self ) -> List[ str ] | None:
		"""Get file purpose options.
		
		Purpose:
			Returns the file purpose options exposed by the Files wrapper. The property centralizes
			UI option values and keeps application selectors aligned with the provider-specific
			implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return [
				'assistants',
				'assistants_output',
				'batch',
				'batch_output',
				'fine-tune',
				'fine-tune-results',
				'vision',
				'user_data',
				'evals',
		]
	
	@property
	def purpose_options( self ) -> List[ str ] | None:
		"""Get purpose options.
		
		Purpose:
			Returns the purpose options exposed by the Files wrapper. The property centralizes UI
			option values and keeps application selectors aligned with the provider-specific
			implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return self.upload_purpose_options
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Get model options.
		
		Purpose:
			Returns the model options exposed by the Files wrapper. The property centralizes UI
			option values and keeps application selectors aligned with the provider-specific
			implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return [
				'gpt-5-mini',
				'gpt-5-nano',
				'gpt-4.1-mini',
				'gpt-4.1-nano',
				'gpt-4o-mini',
		]
	
	def validate_file_id( self, id: str = None ) -> str:
		"""Validate file id.
		
		Purpose:
			Validates and normalizes the file id value used for the Files workflow. The method
			raises an application error when required input is missing and returns a clean value
			suitable for downstream provider calls.
		
		Args:
			id (str): Id value used by the operation.
		
		Returns:
			Validated and normalized value for downstream use.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			value = id if isinstance( id, str ) and id.strip( ) else self.file_id
			throw_if( 'id', value )
			return value.strip( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'validate_file_id( self, id: str=None ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def normalize_file_object( self, file: Any ) -> Dict[ str, Any ]:
		"""Normalize file object.
		
		Purpose:
			Normalizes the file object value used for the Files workflow. The method converts
			provider-specific objects, dictionaries, or compatibility inputs into a stable structure
			for application use.
		
		Args:
			file (Any): File value used by the operation.
		
		Returns:
			Normalized application-facing value or structure.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			if file is None:
				return { }
			
			if isinstance( file, dict ):
				source = file
			elif hasattr( file, 'model_dump' ):
				source = file.model_dump( )
			else:
				source = {
						'id': getattr( file, 'id', None ),
						'bytes': getattr( file, 'bytes', None ),
						'created_at': getattr( file, 'created_at', None ),
						'expires_at': getattr( file, 'expires_at', None ),
						'filename': getattr( file, 'filename', None ),
						'object': getattr( file, 'object', None ),
						'purpose': getattr( file, 'purpose', None ),
						'status': getattr( file, 'status', None ),
						'status_details': getattr( file, 'status_details', None ),
				}
			
			return {
					'id': source.get( 'id' ),
					'filename': source.get( 'filename' ),
					'purpose': source.get( 'purpose' ),
					'bytes': source.get( 'bytes' ),
					'created_at': source.get( 'created_at' ),
					'expires_at': source.get( 'expires_at' ),
					'object': source.get( 'object' ),
					'status': source.get( 'status' ),
					'status_details': source.get( 'status_details' ),
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'normalize_file_object( self, file: Any ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def normalize_file_list( self, response: Any, purpose: str = None ) -> List[ Dict[ str, Any ] ]:
		"""Normalize file list.
		
		Purpose:
			Normalizes the file list value used for the Files workflow. The method converts
			provider-specific objects, dictionaries, or compatibility inputs into a stable structure
			for application use.
		
		Args:
			response (Any): Response value used by the operation.
			purpose (str): Purpose value used by the operation.
		
		Returns:
			Normalized application-facing value or structure.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			if response is None:
				return [ ]
			
			if isinstance( response, list ):
				items = response
			elif isinstance( response, dict ):
				items = response.get( 'data', [ ] )
			else:
				items = getattr( response, 'data', [ ] )
			
			rows: List[ Dict[ str, Any ] ] = [ ]
			for item in items:
				row = self.normalize_file_object( item )
				
				if not row.get( 'id' ):
					continue
				
				if isinstance( purpose, str ) and purpose.strip( ):
					if row.get( 'purpose' ) != purpose.strip( ):
						continue
				
				rows.append( row )
			
			return rows
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'normalize_file_list( self, response: Any, purpose: str=None )'
			Logger( ).write( exception )
			raise exception
	
	def normalize_file_content( self, content: Any ) -> str | bytes | Dict[ str, Any ] | None:
		"""Normalize file content.
		
		Purpose:
			Normalizes the file content value used for the Files workflow. The method converts
			provider-specific objects, dictionaries, or compatibility inputs into a stable structure
			for application use.
		
		Args:
			content (Any): Content value used by the operation.
		
		Returns:
			Normalized application-facing value or structure.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			if content is None:
				return None
			
			if isinstance( content, (str, bytes, dict) ):
				return content
			
			if hasattr( content, 'read' ):
				value = content.read( )
				if isinstance( value, bytes ):
					try:
						return value.decode( 'utf-8' )
					except Exception as e:
						exception = Error( e )
						exception.module = 'gpt'
						exception.cause = 'Files'
						exception.method = 'normalize_file_content( ... )'
						Logger( ).write( exception )
						return value
				
				return value
			
			if hasattr( content, 'text' ):
				value = getattr( content, 'text' )
				if isinstance( value, str ):
					return value
			
			if hasattr( content, 'content' ):
				value = getattr( content, 'content' )
				if isinstance( value, bytes ):
					try:
						return value.decode( 'utf-8' )
					except Exception as e:
						exception = Error( e )
						exception.module = 'gpt'
						exception.cause = 'Files'
						exception.method = 'normalize_file_content( ... )'
						Logger( ).write( exception )
						return value
				
				return value
			
			if hasattr( content, 'model_dump' ):
				return content.model_dump( )
			
			return str( content )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'normalize_file_content( self, content: Any )'
			Logger( ).write( exception )
			raise exception
	
	def upload( self, filepath: str, purpose: str = 'user_data' ) -> Dict[ str, Any ] | None:
		"""Upload.
		
		Purpose:
			Uploads a local file to the OpenAI Files API using a validated purpose value. The method
			stores returned metadata for later retrieval and returns normalized file details.
		
		Args:
			filepath (str): Filepath value used by the operation.
			purpose (str): Purpose value used by the operation.
		
		Returns:
			Normalized provider result when available.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			throw_if( 'filepath', filepath )
			throw_if( 'purpose', purpose )
			
			if not os.path.exists( filepath ):
				raise FileNotFoundError( f'File not found: {filepath}' )
			
			self.client = OpenAI( api_key=self.api_key )
			self.filepath = filepath
			self.purpose = purpose.strip( ) if isinstance( purpose, str ) else purpose
			self.request = { 'file': filepath, 'purpose': self.purpose, }
			with open( self.filepath, 'rb' ) as source:
				self.response = self.client.files.create( file=source, purpose=self.purpose )
			
			self.file = self.response
			metadata = self.normalize_file_object( self.response )
			self.file_id = metadata.get( 'id' )
			self.filename = metadata.get( 'filename' )
			return metadata
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'upload( self, filepath: str, purpose: str )'
			Logger( ).write( exception )
			raise exception
	
	def list( self, purpose: str = None ) -> List[ Dict[ str, Any ] ]:
		"""List.
		
		Purpose:
			Lists provider resources for the Files workflow and returns normalized metadata rows
			suitable for display or follow-on processing.
		
		Args:
			purpose (str): Purpose value used by the operation.
		
		Returns:
			Normalized metadata rows returned by the provider.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			self.client = OpenAI( api_key=self.api_key )
			self.purpose = purpose.strip( ) if isinstance( purpose,
				str ) and purpose.strip( ) else None
			self.request = { }
			self.response = self.client.files.list( )
			self.files = self.normalize_file_list( self.response, purpose=self.purpose )
			return self.files
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'list( self, purpose: str=None ) -> List[ Dict[ str, Any ] ]'
			Logger( ).write( exception )
			raise exception
	
	def retrieve( self, id: str ) -> Dict[ str, Any ] | None:
		"""Retrieve.
		
		Purpose:
			Retrieves a selected provider resource for the Files workflow and returns normalized
			metadata for application use.
		
		Args:
			id (str): Id value used by the operation.
		
		Returns:
			Normalized provider result when available.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			self.client = OpenAI( api_key=self.api_key )
			self.file_id = self.validate_file_id( id )
			self.request = { 'file_id': self.file_id, }
			
			self.response = self.client.files.retrieve( file_id=self.file_id )
			self.file = self.response
			metadata = self.normalize_file_object( self.response )
			self.filename = metadata.get( 'filename' )
			return metadata
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'retrieve( self, id: str ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def extract( self, id: str ) -> str | bytes | Dict[ str, Any ] | None:
		"""Extract.
		
		Purpose:
			Retrieves file content from the OpenAI Files API and normalizes the response into text,
			bytes, or a serializable dictionary.
		
		Args:
			id (str): Id value used by the operation.
		
		Returns:
			Normalized provider result when available.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			self.client = OpenAI( api_key=self.api_key )
			self.file_id = self.validate_file_id( id )
			self.request = { 'file_id': self.file_id, }
			self.response = self.client.files.content( file_id=self.file_id )
			self.content = self.normalize_file_content( self.response )
			return self.content
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'extract( self, id: str )'
			Logger( ).write( exception )
			raise exception
	
	def delete( self, id: str ) -> Dict[ str, Any ] | None:
		"""Delete.
		
		Purpose:
			Deletes a selected provider resource for the Files workflow and returns the provider
			deletion result in a normalized form.
		
		Args:
			id (str): Id value used by the operation.
		
		Returns:
			Normalized provider result when available.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			self.client = OpenAI( api_key=self.api_key )
			self.file_id = self.validate_file_id( id )
			self.request = {
					'file_id': self.file_id,
			}
			
			self.response = self.client.files.delete( file_id=self.file_id )
			
			if isinstance( self.response, dict ):
				return self.response
			
			if hasattr( self.response, 'model_dump' ):
				return self.response.model_dump( )
			
			return {
					'id': getattr( self.response, 'id', self.file_id ),
					'deleted': getattr( self.response, 'deleted', None ),
					'object': getattr( self.response, 'object', None ),
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'delete( self, id: str ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def summarize( self, id: str, prompt: str = None, model: str = 'gpt-4o-mini',
			max_chars: int = 120000 ) -> str | None:
		"""Summarize.
		
		Purpose:
			Summarizes or analyzes retrieved file content with a Responses API model. The method
			extracts file content, limits request size, and returns the generated text response.
		
		Args:
			id (str): Id value used by the operation.
			prompt (str): Prompt value used by the operation.
			model (str): Model value used by the operation.
			max_chars (int): Max chars value used by the operation.
		
		Returns:
			Generated or extracted text when available.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			throw_if( 'model', model )
			
			self.file_id = self.validate_file_id( id )
			self.prompt = prompt if isinstance( prompt, str ) and prompt.strip( ) else \
				'Summarize the selected file content.'
			self.model = model
			content = self.extract( self.file_id )
			
			if isinstance( content, bytes ):
				try:
					content_text = content.decode( 'utf-8' )
				except Exception as e:
					exception = Error( e )
					exception.module = 'gpt'
					exception.cause = 'Files'
					exception.method = 'summarize( ... )'
					Logger( ).write( exception )
					content_text = str( content )
			elif isinstance( content, dict ):
				content_text = str( content )
			else:
				content_text = content if isinstance( content, str ) else ''
			
			throw_if( 'content_text', content_text )
			
			if isinstance( max_chars, int ) and max_chars > 0:
				content_text = content_text[ :max_chars ]
			
			self.client = OpenAI( api_key=self.api_key )
			self.request = { 'model': self.model, 'input': [ {
					'role': 'user',
					'content': [
							{
									'type': 'input_text',
									'text': f'{self.prompt}\n\nFile ID: {self.file_id}\n\n{content_text}',
							}, ],
			}, ], }
			
			self.response = self.client.responses.create( **self.request )
			self.output_text = getattr( self.response, 'output_text', None )
			
			if self.output_text:
				return self.output_text
			
			return str( self.response )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'summarize( self, id: str, prompt: str=None ) -> str | None'
			Logger( ).write( exception )
			raise exception
	
	def search( self, id: str, query: str, model: str = 'gpt-4o-mini',
			max_chars: int = 120000 ) -> str | None:
		"""Search.
		
		Purpose:
			Searches provider-managed content for the Files workflow using a validated query and
			returns normalized search or answer results.
		
		Args:
			id (str): Id value used by the operation.
			query (str): Query value used by the operation.
			model (str): Model value used by the operation.
			max_chars (int): Max chars value used by the operation.
		
		Returns:
			Generated or extracted text when available.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			throw_if( 'query', query )
			throw_if( 'model', model )
			self.prompt = (
				'Answer the user question using the selected file content when possible. '
				f'Question: {query}')
			
			return self.summarize(
				id=id,
				prompt=self.prompt,
				model=model,
				max_chars=max_chars )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'search( self, id: str, query: str ) -> str | None'
			Logger( ).write( exception )
			raise exception
	
	def survey( self, id: str, max_chars: int = 4000 ) -> Dict[ str, Any ]:
		"""Survey.
		
		Purpose:
			Collects metadata, previews, or file-search output for the Files workflow and returns a
			compact application-facing result.
		
		Args:
			id (str): Id value used by the operation.
			max_chars (int): Max chars value used by the operation.
		
		Returns:
			Generated or extracted text when available.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			self.file_id = self.validate_file_id( id )
			metadata = self.retrieve( self.file_id )
			content = self.extract( self.file_id )
			if isinstance( content, bytes ):
				try:
					content_text = content.decode( 'utf-8' )
				except Exception as e:
					exception = Error( e )
					exception.module = 'gpt'
					exception.cause = 'Files'
					exception.method = 'survey( ... )'
					Logger( ).write( exception )
					content_text = str( content )
			elif isinstance( content, dict ):
				content_text = str( content )
			else:
				content_text = content if isinstance( content, str ) else ''
			
			preview = content_text
			if isinstance( max_chars, int ) and max_chars > 0:
				preview = content_text[ :max_chars ]
			
			return { 'metadata': metadata, 'preview': preview, 'file_id': self.file_id, }
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'Files'
			exception.method = 'survey( self, id: str ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def __dir__( self ) -> List[ str ] | None:
		"""Return member names.
		
		Purpose:
			Returns a stable list of public members exposed by the Files object for interactive
			inspection, debugging, and application-level compatibility.
		
		Returns:
			Member names exposed for inspection.
		"""
		return [
				'api_key',
				'client',
				'file',
				'file_id',
				'filepath',
				'filename',
				'purpose',
				'response',
				'content',
				'files',
				'request',
				'model',
				'prompt',
				'output_text',
				'upload_purpose_options',
				'file_purpose_options',
				'purpose_options',
				'model_options',
				'validate_file_id',
				'normalize_file_object',
				'normalize_file_list',
				'normalize_file_content',
				'upload',
				'list',
				'retrieve',
				'extract',
				'delete',
				'summarize',
				'search',
				'survey',
		]

class VectorStores( GPT ):
	"""Provide VectorStores workflow support.
	
	Purpose:
		Provides OpenAI Vector Stores API support for store management, attached-file
		management, file batches, native vector-store search, and Responses API file_search
		workflows.
	
	Attributes:
		api_key (Optional[str]): Api key retained by the provider wrapper.
		client (Optional[OpenAI]): Client retained by the provider wrapper.
		name (Optional[str]): Name retained by the provider wrapper.
		description (Optional[str]): Description retained by the provider wrapper.
		store_id (Optional[str]): Store id retained by the provider wrapper.
		file_id (Optional[str]): File id retained by the provider wrapper.
		batch_id (Optional[str]): Batch id retained by the provider wrapper.
		response (Optional[Any]): Response retained by the provider wrapper.
		vector_store (Optional[Dict[str, Any]]): Vector store retained by the provider wrapper.
		vector_stores (Optional[List[Dict[str, Any]]]): Vector stores retained by the provider wrapper.
		vector_file (Optional[Dict[str, Any]]): Vector file retained by the provider wrapper.
		vector_files (Optional[List[Dict[str, Any]]]): Vector files retained by the provider wrapper.
		file_batch (Optional[Dict[str, Any]]): File batch retained by the provider wrapper.
		search_results (Optional[List[Dict[str, Any]]]): Search results retained by the provider wrapper.
		output_text (Optional[str]): Output text retained by the provider wrapper.
		request (Optional[Dict[str, Any]]): Request retained by the provider wrapper.
		collections (Optional[Dict[str, str]]): Collections retained by the provider wrapper.
		max_search_results (Optional[int]): Max search results retained by the provider wrapper.
	"""
	api_key: Optional[ str ]
	client: Optional[ OpenAI ]
	name: Optional[ str ]
	description: Optional[ str ]
	store_id: Optional[ str ]
	file_id: Optional[ str ]
	batch_id: Optional[ str ]
	response: Optional[ Any ]
	vector_store: Optional[ Dict[ str, Any ] ]
	vector_stores: Optional[ List[ Dict[ str, Any ] ] ]
	vector_file: Optional[ Dict[ str, Any ] ]
	vector_files: Optional[ List[ Dict[ str, Any ] ] ]
	file_batch: Optional[ Dict[ str, Any ] ]
	search_results: Optional[ List[ Dict[ str, Any ] ] ]
	output_text: Optional[ str ]
	request: Optional[ Dict[ str, Any ] ]
	collections: Optional[ Dict[ str, str ] ]
	max_search_results: Optional[ int ]
	
	def __init__( self, name: str = None, store_id: str = None, file_id: str = None,
			model: str = 'gpt-4o-mini', max_search_results: int = 10 ):
		"""Initialize instance.
		
		Purpose:
			Initializes the VectorStores object with default configuration, runtime state, provider
			settings, and compatibility fields. This constructor prepares the instance for later
			method calls without performing external work beyond local attribute assignment.
		
		Args:
			name (str): Name value used by the operation.
			store_id (str): Store id value used by the operation.
			file_id (str): File id value used by the operation.
			model (str): Model value used by the operation.
			max_search_results (int): Max search results value used by the operation.
		"""
		super( ).__init__( )
		self.api_key = cfg.OPENAI_API_KEY
		self.client = None
		self.name = name
		self.description = None
		self.store_id = store_id
		self.file_id = file_id
		self.batch_id = None
		self.model = model
		self.response = None
		self.vector_store = None
		self.vector_stores = [ ]
		self.vector_file = None
		self.vector_files = [ ]
		self.file_batch = None
		self.search_results = [ ]
		self.output_text = None
		self.request = None
		self.max_search_results = max_search_results
		self.collections = {
				'Governance': 'vs_6a1850a9bdc08191912353eedf59aede',
				'Public Laws': 'vs_699506f7d5348191990e0557c717fa9d',
				'Explanatory Statements': 'vs_699505df9ac48191a525c0ecb86fef66',
				'Army Techniques Publications': 'vs_699356ef052c81918da14c4ed3bcea17',
				'Army Field Manuals': 'vs_69935542863481918d150c1e89c38633',
				'Army Regulations': 'vs_6993550488408191919cd70968ba8be8',
				'DoD Armory': 'vs_697f86ad98888191b967685ae558bfc0',
				'Army Style Guides': 'vs_68f4efd7d4c4819191458dd6cde6f2cc',
				'Apportionments': 'vs_68a34aaff93481918c3b3fef8c4e8fea',
				'Financial Regulations': 'vs_712r5W5833G6aLxIYIbuvVcK' }
	
	@property
	def model_options( self ) -> List[ str ] | None:
		"""Get model options.
		
		Purpose:
			Returns the model options exposed by the VectorStores wrapper. The property centralizes
			UI option values and keeps application selectors aligned with the provider-specific
			implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return [
				'gpt-5-mini',
				'gpt-5-nano',
				'gpt-4.1-mini',
				'gpt-4.1-nano',
				'gpt-4o-mini',
		]
	
	@property
	def ranker_options( self ) -> List[ str ] | None:
		"""Get ranker options.
		
		Purpose:
			Returns the ranker options exposed by the VectorStores wrapper. The property centralizes
			UI option values and keeps application selectors aligned with the provider-specific
			implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return [
				'auto',
				'default-2024-11-15',
		]
	
	@property
	def chunking_strategy_options( self ) -> List[ str ] | None:
		"""Get chunking strategy options.
		
		Purpose:
			Returns the chunking strategy options exposed by the VectorStores wrapper. The property
			centralizes UI option values and keeps application selectors aligned with the
			provider-specific implementation.
		
		Returns:
			Available option values exposed by the provider wrapper.
		"""
		return [
				'auto',
				'static',
		]
	
	def validate_store_name( self, name: str = None ) -> str:
		"""Validate store name.
		
		Purpose:
			Validates and normalizes the store name value used for the VectorStores workflow. The
			method raises an application error when required input is missing and returns a clean
			value suitable for downstream provider calls.
		
		Args:
			name (str): Name value used by the operation.
		
		Returns:
			Validated and normalized value for downstream use.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			value = name if isinstance( name, str ) and name.strip( ) else self.name
			throw_if( 'name', value )
			return value.strip( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'validate_store_name( self, name: str=None ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def validate_store_id( self, store_id: str = None ) -> str:
		"""Validate store id.
		
		Purpose:
			Validates and normalizes the store id value used for the VectorStores workflow. The
			method raises an application error when required input is missing and returns a clean
			value suitable for downstream provider calls.
		
		Args:
			store_id (str): Store id value used by the operation.
		
		Returns:
			Validated and normalized value for downstream use.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			value = store_id if isinstance( store_id, str ) and store_id.strip( ) else self.store_id
			throw_if( 'store_id', value )
			return value.strip( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'validate_store_id( self, store_id: str=None ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def validate_file_id( self, file_id: str = None ) -> str:
		"""Validate file id.
		
		Purpose:
			Validates and normalizes the file id value used for the VectorStores workflow. The
			method raises an application error when required input is missing and returns a clean
			value suitable for downstream provider calls.
		
		Args:
			file_id (str): File id value used by the operation.
		
		Returns:
			Validated and normalized value for downstream use.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			value = file_id if isinstance( file_id, str ) and file_id.strip( ) else self.file_id
			throw_if( 'file_id', value )
			return value.strip( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'validate_file_id( self, file_id: str=None ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def validate_batch_id( self, batch_id: str = None ) -> str:
		"""Validate batch id.
		
		Purpose:
			Validates and normalizes the batch id value used for the VectorStores workflow. The
			method raises an application error when required input is missing and returns a clean
			value suitable for downstream provider calls.
		
		Args:
			batch_id (str): Batch id value used by the operation.
		
		Returns:
			Validated and normalized value for downstream use.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			value = batch_id if isinstance( batch_id, str ) and batch_id.strip( ) else self.batch_id
			throw_if( 'batch_id', value )
			return value.strip( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'validate_batch_id( self, batch_id: str=None ) -> str'
			Logger( ).write( exception )
			raise exception
	
	def validate_file_ids( self, file_ids: List[ str ] = None ) -> List[ str ]:
		"""Validate file ids.
		
		Purpose:
			Validates and normalizes the file ids value used for the VectorStores workflow. The
			method raises an application error when required input is missing and returns a clean
			value suitable for downstream provider calls.
		
		Args:
			file_ids (List[str]): File ids value used by the operation.
		
		Returns:
			Validated and normalized value for downstream use.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			if file_ids is None:
				return [ ]
			
			values = [ ]
			for item in file_ids:
				if isinstance( item, str ) and item.strip( ):
					values.append( item.strip( ) )
			
			return values
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'validate_file_ids( self, file_ids: List[ str ]=None )'
			Logger( ).write( exception )
			raise exception
	
	def validate_max_num_results( self, max_num_results: int = None ) -> int:
		"""Validate max num results.
		
		Purpose:
			Validates and normalizes the max num results value used for the VectorStores workflow.
			The method raises an application error when required input is missing and returns a
			clean value suitable for downstream provider calls.
		
		Args:
			max_num_results (int): Max num results value used by the operation.
		
		Returns:
			Validated and normalized value for downstream use.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			value = self.max_search_results if max_num_results is None else int( max_num_results )
			
			if value < 1:
				return 1
			
			if value > 50:
				return 50
			
			return value
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'validate_max_num_results( self, max_num_results: int=None )'
			Logger( ).write( exception )
			raise exception
	
	def build_expires_after( self, anchor: str = None, days: int = None ) -> Dict[
		                                                                         str, Any ] | None:
		"""Build expires after.
		
		Purpose:
			Builds the expires after payload used for the VectorStores workflow. The method
			validates caller input, applies compatibility defaults, and returns a provider-ready
			structure without executing the provider request.
		
		Args:
			anchor (str): Anchor value used by the operation.
			days (int): Days value used by the operation.
		
		Returns:
			Provider-ready request structure or omitted optional payload.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			if days is None:
				return None
			
			value = int( days )
			if value <= 0:
				return None
			
			anchor_value = anchor if isinstance( anchor,
				str ) and anchor.strip( ) else 'last_active_at'
			
			return {
					'anchor': anchor_value.strip( ),
					'days': value,
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'build_expires_after( self, anchor: str=None, days: int=None )'
			Logger( ).write( exception )
			raise exception
	
	def build_chunking_strategy( self, strategy: str = 'auto', max_chunk_size_tokens: int = None,
			chunk_overlap_tokens: int = None ) -> Dict[ str, Any ] | None:
		"""Build chunking strategy.
		
		Purpose:
			Builds the chunking strategy payload used for the VectorStores workflow. The method
			validates caller input, applies compatibility defaults, and returns a provider-ready
			structure without executing the provider request.
		
		Args:
			strategy (str): Strategy value used by the operation.
			max_chunk_size_tokens (int): Max chunk size tokens value used by the operation.
			chunk_overlap_tokens (int): Chunk overlap tokens value used by the operation.
		
		Returns:
			Provider-ready request structure or omitted optional payload.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			strategy_value = strategy if isinstance( strategy,
				str ) and strategy.strip( ) else 'auto'
			strategy_value = strategy_value.strip( )
			
			if strategy_value == 'auto':
				return { 'type': 'auto', }
			
			if strategy_value != 'static':
				return None
			
			max_value = 800 if max_chunk_size_tokens is None else int( max_chunk_size_tokens )
			overlap_value = 400 if chunk_overlap_tokens is None else int( chunk_overlap_tokens )
			
			if max_value < 100:
				max_value = 100
			
			if max_value > 4096:
				max_value = 4096
			
			if overlap_value < 0:
				overlap_value = 0
			
			if overlap_value > max_value // 2:
				overlap_value = max_value // 2
			
			return {
					'type': 'static',
					'static': {
							'max_chunk_size_tokens': max_value,
							'chunk_overlap_tokens': overlap_value,
					},
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'build_chunking_strategy( self, strategy: str, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def normalize_vector_store( self, store: Any ) -> Dict[ str, Any ]:
		"""Normalize vector store.
		
		Purpose:
			Normalizes the vector store value used for the VectorStores workflow. The method
			converts provider-specific objects, dictionaries, or compatibility inputs into a stable
			structure for application use.
		
		Args:
			store (Any): Store value used by the operation.
		
		Returns:
			Normalized application-facing value or structure.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			if store is None:
				return { }
			
			if isinstance( store, dict ):
				source = store
			elif hasattr( store, 'model_dump' ):
				source = store.model_dump( )
			else:
				source = {
						'id': getattr( store, 'id', None ),
						'name': getattr( store, 'name', None ),
						'description': getattr( store, 'description', None ),
						'created_at': getattr( store, 'created_at', None ),
						'object': getattr( store, 'object', None ),
						'usage_bytes': getattr( store, 'usage_bytes', None ),
						'file_counts': getattr( store, 'file_counts', None ),
						'status': getattr( store, 'status', None ),
						'expires_after': getattr( store, 'expires_after', None ),
						'expires_at': getattr( store, 'expires_at', None ),
						'last_active_at': getattr( store, 'last_active_at', None ),
						'metadata': getattr( store, 'metadata', None ),
				}
			
			return {
					'id': source.get( 'id' ),
					'name': source.get( 'name' ),
					'description': source.get( 'description' ),
					'created_at': source.get( 'created_at' ),
					'object': source.get( 'object' ),
					'usage_bytes': source.get( 'usage_bytes' ),
					'file_counts': source.get( 'file_counts' ),
					'status': source.get( 'status' ),
					'expires_after': source.get( 'expires_after' ),
					'expires_at': source.get( 'expires_at' ),
					'last_active_at': source.get( 'last_active_at' ),
					'metadata': source.get( 'metadata' ),
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'normalize_vector_store( self, store: Any ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def normalize_vector_store_file( self, file: Any ) -> Dict[ str, Any ]:
		"""Normalize vector store file.
		
		Purpose:
			Normalizes the vector store file value used for the VectorStores workflow. The method
			converts provider-specific objects, dictionaries, or compatibility inputs into a stable
			structure for application use.
		
		Args:
			file (Any): File value used by the operation.
		
		Returns:
			Normalized application-facing value or structure.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			if file is None:
				return { }
			
			if isinstance( file, dict ):
				source = file
			elif hasattr( file, 'model_dump' ):
				source = file.model_dump( )
			else:
				source = {
						'id': getattr( file, 'id', None ),
						'object': getattr( file, 'object', None ),
						'created_at': getattr( file, 'created_at', None ),
						'vector_store_id': getattr( file, 'vector_store_id', None ),
						'status': getattr( file, 'status', None ),
						'last_error': getattr( file, 'last_error', None ),
						'chunking_strategy': getattr( file, 'chunking_strategy', None ),
						'attributes': getattr( file, 'attributes', None ),
						'usage_bytes': getattr( file, 'usage_bytes', None ),
				}
			
			return {
					'id': source.get( 'id' ),
					'object': source.get( 'object' ),
					'created_at': source.get( 'created_at' ),
					'vector_store_id': source.get( 'vector_store_id' ),
					'status': source.get( 'status' ),
					'last_error': source.get( 'last_error' ),
					'chunking_strategy': source.get( 'chunking_strategy' ),
					'attributes': source.get( 'attributes' ),
					'usage_bytes': source.get( 'usage_bytes' ),
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'normalize_vector_store_file( self, file: Any )'
			Logger( ).write( exception )
			raise exception
	
	def normalize_file_batch( self, batch: Any ) -> Dict[ str, Any ]:
		"""Normalize file batch.
		
		Purpose:
			Normalizes the file batch value used for the VectorStores workflow. The method converts
			provider-specific objects, dictionaries, or compatibility inputs into a stable structure
			for application use.
		
		Args:
			batch (Any): Batch value used by the operation.
		
		Returns:
			Normalized application-facing value or structure.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			if batch is None:
				return { }
			
			if isinstance( batch, dict ):
				source = batch
			elif hasattr( batch, 'model_dump' ):
				source = batch.model_dump( )
			else:
				source = {
						'id': getattr( batch, 'id', None ),
						'object': getattr( batch, 'object', None ),
						'created_at': getattr( batch, 'created_at', None ),
						'vector_store_id': getattr( batch, 'vector_store_id', None ),
						'status': getattr( batch, 'status', None ),
						'file_counts': getattr( batch, 'file_counts', None ),
				}
			
			return {
					'id': source.get( 'id' ),
					'object': source.get( 'object' ),
					'created_at': source.get( 'created_at' ),
					'vector_store_id': source.get( 'vector_store_id' ),
					'status': source.get( 'status' ),
					'file_counts': source.get( 'file_counts' ),
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'normalize_file_batch( self, batch: Any ) -> Dict[ str, Any ]'
			Logger( ).write( exception )
			raise exception
	
	def normalize_search_results( self, response: Any ) -> List[ Dict[ str, Any ] ]:
		"""Normalize search results.
		
		Purpose:
			Normalizes the search results value used for the VectorStores workflow. The method
			converts provider-specific objects, dictionaries, or compatibility inputs into a stable
			structure for application use.
		
		Args:
			response (Any): Response value used by the operation.
		
		Returns:
			Normalized application-facing value or structure.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			if response is None:
				return [ ]
			
			if isinstance( response, dict ):
				items = response.get( 'data', [ ] )
			elif isinstance( response, list ):
				items = response
			else:
				items = getattr( response, 'data', [ ] )
			
			rows: List[ Dict[ str, Any ] ] = [ ]
			for item in items:
				if isinstance( item, dict ):
					source = item
				elif hasattr( item, 'model_dump' ):
					source = item.model_dump( )
				else:
					source = {
							'file_id': getattr( item, 'file_id', None ),
							'filename': getattr( item, 'filename', None ),
							'score': getattr( item, 'score', None ),
							'attributes': getattr( item, 'attributes', None ),
							'content': getattr( item, 'content', None ),
					}
				
				rows.append(
					{
							'file_id': source.get( 'file_id' ),
							'filename': source.get( 'filename' ),
							'score': source.get( 'score' ),
							'attributes': source.get( 'attributes' ),
							'content': source.get( 'content' ),
					} )
			
			return rows
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'normalize_search_results( self, response: Any )'
			Logger( ).write( exception )
			raise exception
	
	def create( self, name: str, description: str = None, metadata: Dict[ str, Any ] = None,
			expires_after: Dict[ str, Any ] = None, file_ids: List[ str ] = None,
			chunking_strategy: Dict[ str, Any ] = None ) -> Dict[ str, Any ] | None:
		"""Create.
		
		Purpose:
			Creates provider resources or generated outputs for the VectorStores workflow using
			validated request state and provider-specific defaults.
		
		Args:
			name (str): Name value used by the operation.
			description (str): Description value used by the operation.
			metadata (Dict[str, Any]): Metadata value used by the operation.
			expires_after (Dict[str, Any]): Expires after value used by the operation.
			file_ids (List[str]): File ids value used by the operation.
			chunking_strategy (Dict[str, Any]): Chunking strategy value used by the operation.
		
		Returns:
			Normalized provider result when available.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			self.client = OpenAI( api_key=self.api_key )
			self.name = self.validate_store_name( name )
			self.description = description if isinstance( description,
				str ) and description.strip( ) else None
			
			self.request = {
					'name': self.name,
			}
			
			if self.description:
				self.request[ 'description' ] = self.description
			
			if isinstance( metadata, dict ) and len( metadata ) > 0:
				self.request[ 'metadata' ] = metadata
			
			if isinstance( expires_after, dict ) and len( expires_after ) > 0:
				self.request[ 'expires_after' ] = expires_after
			
			clean_file_ids = self.validate_file_ids( file_ids )
			if len( clean_file_ids ) > 0:
				self.request[ 'file_ids' ] = clean_file_ids
			
			if isinstance( chunking_strategy, dict ) and len( chunking_strategy ) > 0:
				self.request[ 'chunking_strategy' ] = chunking_strategy
			
			self.response = self.client.vector_stores.create( **self.request )
			self.vector_store = self.normalize_vector_store( self.response )
			self.store_id = self.vector_store.get( 'id' )
			return self.vector_store
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'create( self, name: str, **kwargs ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def list_stores( self, limit: int = 100, order: str = 'desc',
			after: str = None, before: str = None ) -> List[ Dict[ str, Any ] ]:
		"""List stores.
		
		Purpose:
			Executes the list stores operation for the VectorStores wrapper. The method validates
			required inputs, updates runtime state, and returns the application-facing result
			produced by the operation.
		
		Args:
			limit (int): Limit value used by the operation.
			order (str): Order value used by the operation.
			after (str): After value used by the operation.
			before (str): Before value used by the operation.
		
		Returns:
			Normalized metadata rows returned by the provider.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			self.client = OpenAI( api_key=self.api_key )
			self.request = {
					'limit': limit,
					'order': order,
			}
			
			if isinstance( after, str ) and after.strip( ):
				self.request[ 'after' ] = after.strip( )
			
			if isinstance( before, str ) and before.strip( ):
				self.request[ 'before' ] = before.strip( )
			
			self.response = self.client.vector_stores.list( **self.request )
			items = getattr( self.response, 'data', [ ] )
			self.vector_stores = [ self.normalize_vector_store( item ) for item in items ]
			return self.vector_stores
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'list_stores( self, limit: int=100 )'
			Logger( ).write( exception )
			raise exception
	
	def retrieve( self, store_id: str ) -> Dict[ str, Any ] | None:
		"""Retrieve.
		
		Purpose:
			Retrieves a selected provider resource for the VectorStores workflow and returns
			normalized metadata for application use.
		
		Args:
			store_id (str): Store id value used by the operation.
		
		Returns:
			Normalized provider result when available.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			self.client = OpenAI( api_key=self.api_key )
			self.store_id = self.validate_store_id( store_id )
			self.request = {
					'vector_store_id': self.store_id,
			}
			
			self.response = self.client.vector_stores.retrieve(
				vector_store_id=self.store_id )
			self.vector_store = self.normalize_vector_store( self.response )
			return self.vector_store
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'retrieve( self, store_id: str ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def update( self, store_id: str, name: str = None, description: str = None,
			metadata: Dict[ str, Any ] = None,
			expires_after: Dict[ str, Any ] = None ) -> Dict[ str, Any ] | None:
		"""Update.
		
		Purpose:
			Executes the update operation for the VectorStores wrapper. The method validates
			required inputs, updates runtime state, and returns the application-facing result
			produced by the operation.
		
		Args:
			store_id (str): Store id value used by the operation.
			name (str): Name value used by the operation.
			description (str): Description value used by the operation.
			metadata (Dict[str, Any]): Metadata value used by the operation.
			expires_after (Dict[str, Any]): Expires after value used by the operation.
		
		Returns:
			Application-facing result produced by the operation.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			self.client = OpenAI( api_key=self.api_key )
			self.store_id = self.validate_store_id( store_id )
			self.request = { }
			
			if isinstance( name, str ) and name.strip( ):
				self.request[ 'name' ] = name.strip( )
			
			if isinstance( description, str ) and description.strip( ):
				self.request[ 'description' ] = description.strip( )
			
			if isinstance( metadata, dict ):
				self.request[ 'metadata' ] = metadata
			
			if isinstance( expires_after, dict ) and len( expires_after ) > 0:
				self.request[ 'expires_after' ] = expires_after
			
			if len( self.request ) == 0:
				return self.retrieve( self.store_id )
			
			self.response = self.client.vector_stores.update(
				vector_store_id=self.store_id,
				**self.request )
			
			self.vector_store = self.normalize_vector_store( self.response )
			return self.vector_store
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'update( self, store_id: str, **kwargs )'
			Logger( ).write( exception )
			raise exception
	
	def delete( self, store_id: str ) -> Dict[ str, Any ] | None:
		"""Delete.
		
		Purpose:
			Deletes a selected provider resource for the VectorStores workflow and returns the
			provider deletion result in a normalized form.
		
		Args:
			store_id (str): Store id value used by the operation.
		
		Returns:
			Normalized provider result when available.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			self.client = OpenAI( api_key=self.api_key )
			self.store_id = self.validate_store_id( store_id )
			self.request = {
					'vector_store_id': self.store_id,
			}
			
			self.response = self.client.vector_stores.delete(
				vector_store_id=self.store_id )
			
			if isinstance( self.response, dict ):
				return self.response
			
			if hasattr( self.response, 'model_dump' ):
				return self.response.model_dump( )
			
			return {
					'id': getattr( self.response, 'id', self.store_id ),
					'deleted': getattr( self.response, 'deleted', None ),
					'object': getattr( self.response, 'object', None ),
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'delete( self, store_id: str ) -> Dict[ str, Any ] | None'
			Logger( ).write( exception )
			raise exception
	
	def attach_file( self, store_id: str, file_id: str, attributes: Dict[ str, Any ] = None,
			chunking_strategy: Dict[ str, Any ] = None ) -> Dict[ str, Any ] | None:
		"""Attach file.
		
		Purpose:
			Executes the attach file operation for the VectorStores wrapper. The method validates
			required inputs, updates runtime state, and returns the application-facing result
			produced by the operation.
		
		Args:
			store_id (str): Store id value used by the operation.
			file_id (str): File id value used by the operation.
			attributes (Dict[str, Any]): Attributes value used by the operation.
			chunking_strategy (Dict[str, Any]): Chunking strategy value used by the operation.
		
		Returns:
			Application-facing result produced by the operation.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			self.client = OpenAI( api_key=self.api_key )
			self.store_id = self.validate_store_id( store_id )
			self.file_id = self.validate_file_id( file_id )
			self.request = {
					'file_id': self.file_id,
			}
			
			if isinstance( attributes, dict ) and len( attributes ) > 0:
				self.request[ 'attributes' ] = attributes
			
			if isinstance( chunking_strategy, dict ) and len( chunking_strategy ) > 0:
				self.request[ 'chunking_strategy' ] = chunking_strategy
			
			self.response = self.client.vector_stores.files.create(
				vector_store_id=self.store_id,
				**self.request )
			
			self.vector_file = self.normalize_vector_store_file( self.response )
			return self.vector_file
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'attach_file( self, store_id: str, file_id: str )'
			Logger( ).write( exception )
			raise exception
	
	def list( self, store_id: str, limit: int = 100, order: str = 'desc' ) -> List[
		Dict[ str, Any ] ]:
		"""List.
		
		Purpose:
			Lists provider resources for the VectorStores workflow and returns normalized metadata
			rows suitable for display or follow-on processing.
		
		Args:
			store_id (str): Store id value used by the operation.
			limit (int): Limit value used by the operation.
			order (str): Order value used by the operation.
		
		Returns:
			Normalized metadata rows returned by the provider.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			return self.list_files( store_id=store_id, limit=limit, order=order )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'list( self, store_id: str ) -> List[ Dict[ str, Any ] ]'
			Logger( ).write( exception )
			raise exception
	
	def list_files( self, store_id: str, limit: int = 100,
			order: str = 'desc' ) -> List[ Dict[ str, Any ] ]:
		"""List files.
		
		Purpose:
			Executes the list files operation for the VectorStores wrapper. The method validates
			required inputs, updates runtime state, and returns the application-facing result
			produced by the operation.
		
		Args:
			store_id (str): Store id value used by the operation.
			limit (int): Limit value used by the operation.
			order (str): Order value used by the operation.
		
		Returns:
			Normalized metadata rows returned by the provider.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			self.client = OpenAI( api_key=self.api_key )
			self.store_id = self.validate_store_id( store_id )
			self.request = {
					'limit': limit,
					'order': order,
			}
			
			self.response = self.client.vector_stores.files.list(
				vector_store_id=self.store_id,
				**self.request )
			
			items = getattr( self.response, 'data', [ ] )
			self.vector_files = [ self.normalize_vector_store_file( item ) for item in items ]
			return self.vector_files
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'list_files( self, store_id: str )'
			Logger( ).write( exception )
			raise exception
	
	def retrieve_file( self, store_id: str, file_id: str ) -> Dict[ str, Any ] | None:
		"""Retrieve file.
		
		Purpose:
			Executes the retrieve file operation for the VectorStores wrapper. The method validates
			required inputs, updates runtime state, and returns the application-facing result
			produced by the operation.
		
		Args:
			store_id (str): Store id value used by the operation.
			file_id (str): File id value used by the operation.
		
		Returns:
			Application-facing result produced by the operation.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			self.client = OpenAI( api_key=self.api_key )
			self.store_id = self.validate_store_id( store_id )
			self.file_id = self.validate_file_id( file_id )
			
			self.response = self.client.vector_stores.files.retrieve(
				vector_store_id=self.store_id,
				file_id=self.file_id )
			
			self.vector_file = self.normalize_vector_store_file( self.response )
			return self.vector_file
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'retrieve_file( self, store_id: str, file_id: str )'
			Logger( ).write( exception )
			raise exception
	
	def update_file( self, store_id: str, file_id: str,
			attributes: Dict[ str, Any ] = None ) -> Dict[ str, Any ] | None:
		"""Update file.
		
		Purpose:
			Executes the update file operation for the VectorStores wrapper. The method validates
			required inputs, updates runtime state, and returns the application-facing result
			produced by the operation.
		
		Args:
			store_id (str): Store id value used by the operation.
			file_id (str): File id value used by the operation.
			attributes (Dict[str, Any]): Attributes value used by the operation.
		
		Returns:
			Application-facing result produced by the operation.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			self.client = OpenAI( api_key=self.api_key )
			self.store_id = self.validate_store_id( store_id )
			self.file_id = self.validate_file_id( file_id )
			self.request = { }
			
			if isinstance( attributes, dict ):
				self.request[ 'attributes' ] = attributes
			
			self.response = self.client.vector_stores.files.update(
				vector_store_id=self.store_id,
				file_id=self.file_id,
				**self.request )
			
			self.vector_file = self.normalize_vector_store_file( self.response )
			return self.vector_file
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'update_file( self, store_id: str, file_id: str )'
			Logger( ).write( exception )
			raise exception
	
	def delete_file( self, store_id: str, file_id: str ) -> Dict[ str, Any ] | None:
		"""Delete file.
		
		Purpose:
			Executes the delete file operation for the VectorStores wrapper. The method validates
			required inputs, updates runtime state, and returns the application-facing result
			produced by the operation.
		
		Args:
			store_id (str): Store id value used by the operation.
			file_id (str): File id value used by the operation.
		
		Returns:
			Application-facing result produced by the operation.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			self.client = OpenAI( api_key=self.api_key )
			self.store_id = self.validate_store_id( store_id )
			self.file_id = self.validate_file_id( file_id )
			
			self.response = self.client.vector_stores.files.delete(
				vector_store_id=self.store_id,
				file_id=self.file_id )
			
			if isinstance( self.response, dict ):
				return self.response
			
			if hasattr( self.response, 'model_dump' ):
				return self.response.model_dump( )
			
			return {
					'id': getattr( self.response, 'id', self.file_id ),
					'deleted': getattr( self.response, 'deleted', None ),
					'object': getattr( self.response, 'object', None ),
			}
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'delete_file( self, store_id: str, file_id: str )'
			Logger( ).write( exception )
			raise exception
	
	def retrieve_file_content( self, store_id: str, file_id: str ) -> Any:
		"""Retrieve file content.
		
		Purpose:
			Executes the retrieve file content operation for the VectorStores wrapper. The method
			validates required inputs, updates runtime state, and returns the application-facing
			result produced by the operation.
		
		Args:
			store_id (str): Store id value used by the operation.
			file_id (str): File id value used by the operation.
		
		Returns:
			Application-facing result produced by the operation.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			self.client = OpenAI( api_key=self.api_key )
			self.store_id = self.validate_store_id( store_id )
			self.file_id = self.validate_file_id( file_id )
			
			self.response = self.client.vector_stores.files.content(
				vector_store_id=self.store_id,
				file_id=self.file_id )
			
			return self.response
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'retrieve_file_content( self, store_id: str, file_id: str )'
			Logger( ).write( exception )
			raise exception
	
	def create_file_batch( self, store_id: str, file_ids: List[ str ],
			attributes: Dict[ str, Any ] = None,
			chunking_strategy: Dict[ str, Any ] = None ) -> Dict[ str, Any ] | None:
		"""Create file batch.
		
		Purpose:
			Executes the create file batch operation for the VectorStores wrapper. The method
			validates required inputs, updates runtime state, and returns the application-facing
			result produced by the operation.
		
		Args:
			store_id (str): Store id value used by the operation.
			file_ids (List[str]): File ids value used by the operation.
			attributes (Dict[str, Any]): Attributes value used by the operation.
			chunking_strategy (Dict[str, Any]): Chunking strategy value used by the operation.
		
		Returns:
			Application-facing result produced by the operation.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			self.client = OpenAI( api_key=self.api_key )
			self.store_id = self.validate_store_id( store_id )
			clean_file_ids = self.validate_file_ids( file_ids )
			throw_if( 'file_ids', clean_file_ids )
			
			if len( clean_file_ids ) > 2000:
				raise ValueError( 'Vector store file batches cannot exceed 2000 files.' )
			
			self.request = {
					'file_ids': clean_file_ids,
			}
			
			if isinstance( attributes, dict ) and len( attributes ) > 0:
				self.request[ 'attributes' ] = attributes
			
			if isinstance( chunking_strategy, dict ) and len( chunking_strategy ) > 0:
				self.request[ 'chunking_strategy' ] = chunking_strategy
			
			self.response = self.client.vector_stores.file_batches.create(
				vector_store_id=self.store_id,
				**self.request )
			
			self.file_batch = self.normalize_file_batch( self.response )
			self.batch_id = self.file_batch.get( 'id' )
			return self.file_batch
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'create_file_batch( self, store_id: str, file_ids: List[ str ] )'
			Logger( ).write( exception )
			raise exception
	
	def retrieve_file_batch( self, store_id: str, batch_id: str ) -> Dict[ str, Any ] | None:
		"""Retrieve file batch.
		
		Purpose:
			Executes the retrieve file batch operation for the VectorStores wrapper. The method
			validates required inputs, updates runtime state, and returns the application-facing
			result produced by the operation.
		
		Args:
			store_id (str): Store id value used by the operation.
			batch_id (str): Batch id value used by the operation.
		
		Returns:
			Application-facing result produced by the operation.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			self.client = OpenAI( api_key=self.api_key )
			self.store_id = self.validate_store_id( store_id )
			self.batch_id = self.validate_batch_id( batch_id )
			
			self.response = self.client.vector_stores.file_batches.retrieve(
				vector_store_id=self.store_id,
				batch_id=self.batch_id )
			
			self.file_batch = self.normalize_file_batch( self.response )
			return self.file_batch
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'retrieve_file_batch( self, store_id: str, batch_id: str )'
			Logger( ).write( exception )
			raise exception
	
	def list_file_batch_files( self, store_id: str, batch_id: str,
			limit: int = 100 ) -> List[ Dict[ str, Any ] ]:
		"""List file batch files.
		
		Purpose:
			Executes the list file batch files operation for the VectorStores wrapper. The method
			validates required inputs, updates runtime state, and returns the application-facing
			result produced by the operation.
		
		Args:
			store_id (str): Store id value used by the operation.
			batch_id (str): Batch id value used by the operation.
			limit (int): Limit value used by the operation.
		
		Returns:
			Normalized metadata rows returned by the provider.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			self.client = OpenAI( api_key=self.api_key )
			self.store_id = self.validate_store_id( store_id )
			self.batch_id = self.validate_batch_id( batch_id )
			
			self.response = self.client.vector_stores.file_batches.files.list(
				vector_store_id=self.store_id,
				batch_id=self.batch_id,
				limit=limit )
			
			items = getattr( self.response, 'data', [ ] )
			self.vector_files = [ self.normalize_vector_store_file( item ) for item in items ]
			return self.vector_files
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'list_file_batch_files( self, store_id: str, batch_id: str )'
			Logger( ).write( exception )
			raise exception
	
	def cancel_file_batch( self, store_id: str, batch_id: str ) -> Dict[ str, Any ] | None:
		"""Cancel file batch.
		
		Purpose:
			Executes the cancel file batch operation for the VectorStores wrapper. The method
			validates required inputs, updates runtime state, and returns the application-facing
			result produced by the operation.
		
		Args:
			store_id (str): Store id value used by the operation.
			batch_id (str): Batch id value used by the operation.
		
		Returns:
			Application-facing result produced by the operation.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			self.client = OpenAI( api_key=self.api_key )
			self.store_id = self.validate_store_id( store_id )
			self.batch_id = self.validate_batch_id( batch_id )
			
			self.response = self.client.vector_stores.file_batches.cancel(
				vector_store_id=self.store_id,
				batch_id=self.batch_id )
			
			self.file_batch = self.normalize_file_batch( self.response )
			return self.file_batch
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'cancel_file_batch( self, store_id: str, batch_id: str )'
			Logger( ).write( exception )
			raise exception
	
	def search( self, store_id: str, query: str, max_num_results: int = 10,
			filters: Dict[ str, Any ] = None, ranking_options: Dict[ str, Any ] = None,
			rewrite_query: bool = None ) -> List[ Dict[ str, Any ] ]:
		"""Search.
		
		Purpose:
			Searches provider-managed content for the VectorStores workflow using a validated query
			and returns normalized search or answer results.
		
		Args:
			store_id (str): Store id value used by the operation.
			query (str): Query value used by the operation.
			max_num_results (int): Max num results value used by the operation.
			filters (Dict[str, Any]): Filters value used by the operation.
			ranking_options (Dict[str, Any]): Ranking options value used by the operation.
			rewrite_query (bool): Rewrite query value used by the operation.
		
		Returns:
			Generated or extracted text when available.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			return self.search_store( store_id=store_id, query=query,
				max_num_results=max_num_results,
				filters=filters, ranking_options=ranking_options, rewrite_query=rewrite_query )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'search( self, store_id: str, query: str )'
			Logger( ).write( exception )
			raise exception
	
	def search_store( self, store_id: str, query: str, max_num_results: int = 10,
			filters: Dict[ str, Any ] = None, ranking_options: Dict[ str, Any ] = None,
			rewrite_query: bool = None ) -> List[ Dict[ str, Any ] ]:
		"""Search store.
		
		Purpose:
			Executes the search store operation for the VectorStores wrapper. The method validates
			required inputs, updates runtime state, and returns the application-facing result
			produced by the operation.
		
		Args:
			store_id (str): Store id value used by the operation.
			query (str): Query value used by the operation.
			max_num_results (int): Max num results value used by the operation.
			filters (Dict[str, Any]): Filters value used by the operation.
			ranking_options (Dict[str, Any]): Ranking options value used by the operation.
			rewrite_query (bool): Rewrite query value used by the operation.
		
		Returns:
			Application-facing result produced by the operation.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			self.client = OpenAI( api_key=self.api_key )
			self.store_id = self.validate_store_id( store_id )
			throw_if( 'query', query )
			
			self.request = {
					'query': query.strip( ),
					'max_num_results': self.validate_max_num_results( max_num_results ),
			}
			
			if isinstance( filters, dict ) and len( filters ) > 0:
				self.request[ 'filters' ] = filters
			
			if isinstance( ranking_options, dict ) and len( ranking_options ) > 0:
				self.request[ 'ranking_options' ] = ranking_options
			
			if isinstance( rewrite_query, bool ):
				self.request[ 'rewrite_query' ] = rewrite_query
			
			self.response = self.client.vector_stores.search(
				vector_store_id=self.store_id,
				**self.request )
			
			self.search_results = self.normalize_search_results( self.response )
			return self.search_results
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'search_store( self, store_id: str, query: str )'
			Logger( ).write( exception )
			raise exception
	
	def answer_with_file_search( self, store_ids: List[ str ], prompt: str,
			model: str = 'gpt-4o-mini', max_num_results: int = 10,
			instructions: str = None ) -> str | None:
		"""Answer with file search.
		
		Purpose:
			Answers a user prompt with the Responses API file_search tool across selected vector
			stores. The method validates vector store identifiers and returns the generated answer
			text.
		
		Args:
			store_ids (List[str]): Store ids value used by the operation.
			prompt (str): Prompt value used by the operation.
			model (str): Model value used by the operation.
			max_num_results (int): Max num results value used by the operation.
			instructions (str): Instructions value used by the operation.
		
		Returns:
			Generated or extracted text when available.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			self.client = OpenAI( api_key=self.api_key )
			clean_store_ids = [
					item.strip( ) for item in store_ids
					if isinstance( item, str ) and item.strip( )
			]
			
			throw_if( 'store_ids', clean_store_ids )
			throw_if( 'prompt', prompt )
			
			model_value = model if isinstance( model, str ) and model.strip( ) else 'gpt-4o-mini'
			
			input_items: List[ Dict[ str, Any ] ] = [ ]
			if isinstance( instructions, str ) and instructions.strip( ):
				input_items.append(
					{
							'role': 'developer',
							'content': [
									{
											'type': 'input_text',
											'text': instructions.strip( ),
									}, ],
					} )
			
			input_items.append(
				{
						'role': 'user',
						'content': [
								{
										'type': 'input_text',
										'text': prompt.strip( ),
								}, ],
				} )
			
			self.request = {
					'model': model_value,
					'input': input_items,
					'tools': [
							{
									'type': 'file_search',
									'vector_store_ids': clean_store_ids,
									'max_num_results': self.validate_max_num_results(
										max_num_results ),
							}, ],
			}
			
			self.response = self.client.responses.create( **self.request )
			self.output_text = getattr( self.response, 'output_text', None )
			
			if self.output_text:
				return self.output_text
			
			return str( self.response )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'answer_with_file_search( self, store_ids: List[ str ], prompt: str )'
			Logger( ).write( exception )
			raise exception
	
	def survey( self, store_ids: List[ str ], prompt: str = None, model: str = 'gpt-4o-mini',
			max_num_results: int = 10, instructions: str = None ) -> str | None:
		"""Survey.
		
		Purpose:
			Collects metadata, previews, or file-search output for the VectorStores workflow and
			returns a compact application-facing result.
		
		Args:
			store_ids (List[str]): Store ids value used by the operation.
			prompt (str): Prompt value used by the operation.
			model (str): Model value used by the operation.
			max_num_results (int): Max num results value used by the operation.
			instructions (str): Instructions value used by the operation.
		
		Returns:
			Generated or extracted text when available.
		
		Raises:
			Error: Re-raised after the exception is wrapped and written to the application logger.
		"""
		try:
			query = prompt if isinstance( prompt, str ) and prompt.strip( ) else \
				'Summarize the most relevant information available in the selected vector stores.'
			
			return self.answer_with_file_search(
				store_ids=store_ids,
				prompt=query,
				model=model,
				max_num_results=max_num_results,
				instructions=instructions )
		except Exception as e:
			exception = Error( e )
			exception.module = 'gpt'
			exception.cause = 'VectorStores'
			exception.method = 'survey( self, store_ids: List[ str ], prompt: str=None )'
			Logger( ).write( exception )
			raise exception
	
	def __dir__( self ) -> List[ str ] | None:
		"""Return member names.
		
		Purpose:
			Returns a stable list of public members exposed by the VectorStores object for
			interactive inspection, debugging, and application-level compatibility.
		
		Returns:
			Member names exposed for inspection.
		"""
		return [
				'api_key',
				'client',
				'name',
				'description',
				'store_id',
				'file_id',
				'batch_id',
				'model',
				'response',
				'vector_store',
				'vector_stores',
				'vector_file',
				'vector_files',
				'file_batch',
				'search_results',
				'output_text',
				'request',
				'collections',
				'max_search_results',
				'model_options',
				'ranker_options',
				'chunking_strategy_options',
				'validate_store_name',
				'validate_store_id',
				'validate_file_id',
				'validate_batch_id',
				'validate_file_ids',
				'validate_max_num_results',
				'build_expires_after',
				'build_chunking_strategy',
				'normalize_vector_store',
				'normalize_vector_store_file',
				'normalize_file_batch',
				'normalize_search_results',
				'create',
				'list_stores',
				'retrieve',
				'update',
				'delete',
				'attach_file',
				'list',
				'list_files',
				'retrieve_file',
				'update_file',
				'delete_file',
				'retrieve_file_content',
				'create_file_batch',
				'retrieve_file_batch',
				'list_file_batch_files',
				'cancel_file_batch',
				'search',
				'search_store',
				'answer_with_file_search',
				'survey',
		]
		
		