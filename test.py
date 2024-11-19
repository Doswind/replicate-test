
import gradio as gr  
  
def page1():  
    return gr.Interface(fn=some_fn, inputs="text", outputs="text", title="Page 1")  
  
def page2():  
    return gr.Interface(fn=some_fn2, inputs="text", outputs="text", title="Page 2")  
  
def page3():  
    return gr.Interface(fn=some_fn3, inputs="text", outputs="text", title="Page 3")  
  
def main():  
    page = gr.Interface(fn=page1, inputs="text", outputs="text", title="Page 1")  
    page2 = gr.Interface(fn=page2, inputs="text", outputs="text", title="Page 2")  
    page3 = gr.Interface(fn=page3, inputs="text", outputs="text", title="Page 3")  
    gr.Interface(fn=page1, inputs="text", outputs="text", title="Page 1")  
    gr.Interface(fn=page2, inputs="text", outputs="text", title="Page 2")  
    gr.Interface(fn=page3, inputs="text", outputs="text", title="Page 3")  
    return "End"  
  
def switch_page(choice):  
    if choice == "Page 1":  
        return page1()  
    elif choice == "Page 2":  
        return page2()  
    elif choice == "Page 3":  
        return page3()


if __name__ == "__main__":  
    print(gr.shortcut_main(main))