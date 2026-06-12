{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}

{%- if attributes %}

   .. rubric:: {{ _('Module Attributes') }}

   .. autosummary::
{% for item in attributes %}
      {{ item }}
{%- endfor %}
{%- endif %}
{%- if functions %}

   .. rubric:: {{ _('Functions') }}

   .. autosummary::
{% for item in functions %}
      {{ item }}
{%- endfor %}
{%- endif %}
{%- if classes %}

   .. rubric:: {{ _('Classes') }}

   .. autosummary::
{% for item in classes %}
      {{ item }}
{%- endfor %}
{%- endif %}
{%- if exceptions %}

   .. rubric:: {{ _('Exceptions') }}

   .. autosummary::
{% for item in exceptions %}
      {{ item }}
{%- endfor %}
{%- endif %}
{%- if modules %}

.. rubric:: Modules

.. autosummary::
   :toctree:
   :recursive:
{% for item in modules %}
   {{ item }}
{%- endfor %}
{%- endif %}
