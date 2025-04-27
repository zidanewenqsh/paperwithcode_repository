import pypandoc
# pypandoc.download_pandoc()
# output = pypandoc.convert_file('doublefusion.md', 'pdf', outputfile='doublefusion.pdf')
# print("PDF generated:", output)
# import pypandoc

output = pypandoc.convert_file(
    'doublefusion.md',
    'pdf',
    outputfile='doublefusion.pdf',
    extra_args=['--pdf-engine=wkhtmltopdf']
)
print("PDF generated:", output)
