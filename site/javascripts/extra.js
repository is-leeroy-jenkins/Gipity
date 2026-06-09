document.addEventListener( "DOMContentLoaded", function()
{
	const tables = document.querySelectorAll( "table" );
	tables.forEach( function( table )
	{
		table.setAttribute( "tabindex", "0" );
	} );
} );