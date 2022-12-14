import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import tensorflow.keras as keras


def set_ax(ax, xlabel=None, ylabel=None, title=None,
           fontsize=14, legend=False, grid=False):
    ax.tick_params(labelsize=fontsize)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    if title:
        ax.set_title(title, fontsize=fontsize)
    if legend:
        ax.legend(fontsize=fontsize)
    if grid:
        ax.grid()
        
def pred_binary(model, imgnp, img_name='this', cat0='category 0', cat1='category 1'):
    classes = model.predict(imgnp[np.newaxis, ...])
    
    fig, ax = plt.subplots()
    ax.imshow(imgnp)
    ax.axis('off')
    ax.set_title(
        f'P({cat1}) = {classes[0, 0]:.2g}\n'
        + img_name + ' is a ' + (cat1 if classes[0, 0] >= 0.5 else cat0)
    )

def pred_binary_dir(model, test_dir, cat0='category 0', cat1='category 1'):
    img_names = [file_name for file_name in os.listdir(test_dir)
                           if file_name.endswith(('.png', '.jpg'))]
    for img_name in img_names:
        img = keras.utils.load_img(os.path.join(test_dir, img_name),
                                   target_size=model.input_shape[1:3])
        imgnp = keras.utils.img_to_array(img)/255
        pred_binary(model, imgnp, img_name, cat0, cat1)
        
def pred_categorical(model, imgnp, img_name='this', cats=None):
    classes = model.predict(imgnp[np.newaxis, ...])
    
    fig, ax = plt.subplots()
    ax.imshow(imgnp)
    ax.axis('off')
    ax.set_title(
        f'P({cat1}) = {classes[0, 0]:.2g}\n'
        + img_name + ' is a ' + (cat1 if classes[0, 0] >= 0.5 else cat0)
    )

def pred_dir(model, test_dir, cats):
    img_names = [file_name for file_name in os.listdir(test_dir)
                           if file_name.endswith(('.png', '.jpg'))]
    for img_name in img_names:
        img = keras.utils.load_img(os.path.join(test_dir, img_name),
                                   target_size=model.input_shape[1:3])
        imgnp = keras.utils.img_to_array(img)/255
        if len(cats) == 2:
            pred_binary(model, imgnp, img_name, *cats)
        else:
            pred_categorical(model, imgnp, img_name, cats)
        
def conv_layers(model):
    layers_out = [l.output for l in model.layers if len(l.output_shape) == 4]
    return keras.models.Model(inputs=model.input, outputs=layers_out)

def view_layers_output(layers_model, img, fig_x = 20, n_fil_row = 16):
    out = layers_model.predict(img[np.newaxis, ...])
    
    for l_out, l in zip(out, layers_model.outputs):
        n_fil = l_out.shape[3]
        side = l_out.shape[1]
        
        stacked = np.zeros((side, side*n_fil))
        for i in range(n_fil):
            x = l_out[0, :, :, i]
            x -= x.mean()
            if x.std() > 1e-6:
                x /= x.std()
            x = np.clip(x, -2, 2)
            stacked[:, i*side : (i+1)*side] = x
            
        nrow = n_fil // n_fil_row
        stacked = np.vstack(np.split(stacked, nrow, axis=1))
        
        fig, ax = plt.subplots(figsize=(fig_x, fig_x/n_fil_row*nrow))
        ax.imshow(stacked, cmap='viridis')
        ax.axis('off')
        ax.set_title(l.name + ': ' + str(l.shape))
        
def get_layers_viewer(model, cat0='category 0', cat1='category 1', img_dir=None):
    conv_model = conv_layers(model)
    
    def viewer(img_name, img_dir=img_dir):
        file = os.path.join(img_dir, img_name)
        
        img = keras.utils.load_img(file, target_size=model.input_shape[1:3])
        imgnp = keras.utils.img_to_array(img)/255
        
        pred_binary(model, imgnp, img_name, cat0, cat1)
        view_layers_output(conv_model, imgnp)
    
    return viewer

def plot_train_val_metrics(history, metrics, start=0):
    epochs = np.array(history.epoch)
    
    nmet = len(metrics)
    if nmet % 2 == 0:
        nrow = nmet // 2
        ncol = 2
    else:
        nrow = nmet // 2 + 1
        ncol = 1 if nmet == 1 else 2
    
    fig, axs = plt.subplots(nrow, ncol, figsize=(7*ncol, 4*nrow))
    axs = [axs] if nmet == 1 else axs.flatten()
    
    for i, met in enumerate(metrics):
        trn_min = min(history.history[met])
        axs[i].plot(epochs[start:],
                    history.history[met][start:],
                    label=f'train min={trn_min:.3g}')
        
        if ('val_' + met) in history.history:
            val_min = min(history.history['val_'+met])
            axs[i].plot(epochs[start:],
                        history.history['val_'+met][start:],
                        label=f'val min={val_min:.3g}')
        
        set_ax(axs[i], 'epoch', title=met, legend=True, grid=True)
        
def save_tsv(model, tokenizer, save_dir):
    vecs = save_dir + '-v.tsv'
    meta = save_dir + '-m.tsv'
    
    embedl = model.layers[0]
    embedw = embedl.get_weights()[0]
    
    num_words = embedw.shape[0]
    
    with open(vecs, 'w') as out_v:
        with open(meta, 'w') as out_m:
            for i in range(1, num_words):
                out_v.write('\t'.join(str(x) for x in embedw[i]) + '\n')
                out_m.write(tokenizer.index_word[i] + '\n')
                
def embed_extremes(model, tokenizer, d=0, n=20):
    wts = (model.layers[0].get_weights()[0][1:, d]
           * np.sign(model.layers[2].get_weights()[0][0, d]))
    idx = np.argsort(wts)
    for i in range(n):
        print(f'{wts[idx[i]]:-10.4f} {tokenizer.index_word[idx[i] + 1]:15}'
              f'{wts[idx[-1-i]]:-10.4f} {tokenizer.index_word[idx[-1-i] + 1]:15}')
        
        
class Heap:
    def __init__(self, items, get_value=lambda x: x):
        self.items = list(items)
        self.get_value = get_value
        for i in range(len(self.items))[::-1]:
            self._topdown(i)
        
    def _topdown(self, i):
        l, r = i*2 + 1, i*2 + 2
        if l < len(self.items):
            if ((r >= len(self.items))
                or (self.get_value(self.items[l]) >= self.get_value(self.items[r]))):
                ch = l
            else:
                ch = r
            if self.get_value(self.items[i]) < self.get_value(self.items[ch]):
                self.items[i], self.items[ch] = self.items[ch], self.items[i]
                self._topdown(ch)
            
    def _bottomup(self, i):
        if i > 0:
            pr = (i - 1)//2
            if self.get_value(self.items[i]) > self.get_value(self.items[pr]):
                self.items[i], self.items[pr] = self.items[pr], self.items[i]
                self._bottomup(pr)
                
    def numel(self):
        return len(self.items)
            
    def root(self):
        return self.items[0]
        
    def pop(self):
        root, self.items[0] = self.items[0], self.items[-1]
        del self.items[-1]
        self._topdown(0)
        return root
    
    def insert(self, newitem):
        self.items.append(newitem)
        self._bottomup(len(self.items) - 1)
        
    def _verify(self, i=0):
        l, r = i*2 + 1, i*2 + 2
        if l >= len(self.items):
            return True
        else:
            if self.get_value(self.items[i]) < self.get_value(self.items[l]):
                print(f'heap violation: node {i} < node {l}')
                return False
            else:
                if r >= len(self.items):
                    return self._verify(l)
                else:
                    if self.get_value(self.items[i]) < self.get_value(self.items[r]):
                        print(f'heap violation: node {i} < node {r}')
                        return False
                    else:
                        return self._verify(l) and self._verify(r)
                    