use proc_macro::TokenStream;
use quote::{ToTokens, quote};
use syn::{Expr, Field, ItemStruct, Meta, Type, parse::Parser, parse_macro_input};

#[proc_macro_attribute]
pub fn rust_class(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemStruct);
    let struct_name = &input.ident;
    let struct_name_impl = syn::Ident::new(&format!("{struct_name}Impl"), struct_name.span());
    let visibility = &input.vis;
    let generics = &input.generics;

    // Default superclass (`()`)
    let mut super_class: Type = syn::parse_quote!(());

    // Parse attribute as `super = SomeSuperClass`
    if !attr.is_empty() {
        let meta = parse_macro_input!(attr as Meta);

        match meta {
            // **Handles Full Format** → `super = SomeSuperClass`
            Meta::NameValue(meta_name_value) if meta_name_value.path.is_ident("super") => {
                if let Expr::Path(expr_path) = &meta_name_value.value {
                    super_class = syn::parse_quote!(#expr_path);
                }
            }

            // **Handles Shorthand Format** → `#[rust_class(SomeSuperClass)]`
            Meta::Path(path) => {
                if let Ok(parsed_type) = syn::parse2::<Type>(path.clone().into_token_stream()) {
                    super_class = parsed_type;
                } else {
                    return TokenStream::from(quote! {
                        compile_error!("Failed to parse superclass. Ensure it's a valid type and properly formatted.");
                    });
                }
            }

            _ => {
                return TokenStream::from(quote! {
                    compile_error!("Invalid attribute format. Use `#[rust_class(super = SomeSuperClass)]` or `#[rust_class(SomeSuperClass)]`.");
                });
            }
        }
    }

    // Collect existing fields
    let mut fields = input.fields.clone().into_iter().collect::<Vec<_>>();

    // Add `_super` field dynamically using the parsed `super_class` type
    match Field::parse_named.parse2(quote! { _super: Option<#super_class> }) {
        Ok(field) => fields.push(field),
        Err(_) => {
            return TokenStream::from(quote! {
                compile_error!("Failed to parse superclass. Ensure it's a valid type.");
            });
        }
    }

    // Generate output
    let expanded = quote! {
        #visibility type #struct_name #generics = #struct_name_impl #generics;

        #[derive(Default)]
        #visibility struct #struct_name_impl #generics {
            #(#fields),*
        }

        // Implement `Deref`
        impl #generics core::ops::Deref for #struct_name_impl #generics {
            type Target = #super_class;

            fn deref(&self) -> &Self::Target {
                self._super.as_ref().expect("Superclass is not set!")
            }
        }

        // Implement `DerefMut`
        impl #generics core::ops::DerefMut for #struct_name_impl #generics {
            fn deref_mut(&mut self) -> &mut Self::Target {
                self._super.as_mut().expect("Superclass is not set!")
            }
        }
    };

    TokenStream::from(expanded)
}
