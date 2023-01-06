# Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models

We show the results of the text-to-3D task supported by the [DreamFusion Project](https://github.com/ashawkey/stable-dreamfusion).

## Usage of Adan for DreamFusion

Adan is the default optimizer for the [DreamFusion Project](https://github.com/ashawkey/stable-dreamfusion); please refer to its repo to run these experiments.

The project calls the Adan as follows:

```
optimizer = lambda model: Adan(model.get_params(5 * opt.lr), eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False)
```

We may tune learning rate `opt.lr` and maximal gradient norm `max_grad_norm` to refine the results w.r.t. some text prompts.

## Training and Evaluation

- #### Training

  ` python main.py --text $PROMPT --workspace $SAVE_PATH -O`

- #### Evaluation

  `python main.py --workspace $SAVE_PATH -O --test`

## Results

**prompt:** `a DSLR photo of the leaning tower of Pisa, aerial view`. Adan‘s model has more refined details.

https://user-images.githubusercontent.com/10042844/211014605-3860b816-cc1c-4367-b96e-406cd375240a.mp4

https://user-images.githubusercontent.com/10042844/211014603-82564238-cf5b-4ffa-b7a3-175bd565e5ce.mp4

**prompt:** `Sydney opera house, aerial view`. Adan provides better details.

https://user-images.githubusercontent.com/10042844/211014601-da430196-021d-4f6b-962b-8441feff5d02.mp4

https://user-images.githubusercontent.com/10042844/211014594-3b5c05e3-9018-4a39-b5db-d6f2fc111cce.mp4

**prompt:** `the Statue of Liberty, aerial view`. Adan has a better picture with this prompt.

https://user-images.githubusercontent.com/10042844/211014579-4db62a55-fd05-4616-9793-5af5fea81c62.mp4

https://user-images.githubusercontent.com/10042844/211014575-db8b9b1b-7e81-4a27-ba36-2ef74c00f0bc.mp4

**prompt:** `the Imperial State Crown of England`

https://user-images.githubusercontent.com/10042844/211014561-7a943df3-ed8f-4c1a-b51f-8ca5bccf1819.mp4

https://user-images.githubusercontent.com/10042844/211014554-b7f696dd-8635-4d75-81c3-218dd0231c76.mp4

**prompt:** `a candelabra with many candles`. Adam's model has some candles suspended in the air while Adan's result is more clear.

https://user-images.githubusercontent.com/10042844/211014542-47f19116-9fb9-4e65-ad08-522d1c97ba11.mp4

https://user-images.githubusercontent.com/10042844/211014532-6dec1554-c552-4fc5-92c4-cf9954d844cb.mp4

**prompt:** `an extravagant mansion, aerial view`. Adan's result is more meaningful.

https://user-images.githubusercontent.com/10042844/211014591-82d6e57e-bc9f-4b38-8d23-9b156a35334c.mp4

https://user-images.githubusercontent.com/10042844/211014584-aa038ea9-58ae-422f-a128-e885d7d7ab08.mp4

**prompt:** `Neuschwanstein Castle, aerial view`

https://user-images.githubusercontent.com/10042844/211014548-160c7416-d74f-48aa-b3dc-bfd55e809b62.mp4

https://user-images.githubusercontent.com/10042844/211014545-2515b2be-bff8-4e7c-9718-0ee0210c98e9.mp4

**prompt:** `a delicious hamburger`

https://user-images.githubusercontent.com/10042844/211014566-ae9c6f72-2bbf-4e4b-8f15-27851464a620.mp4

https://user-images.githubusercontent.com/10042844/211014571-af207d24-1119-4b34-a31d-5250046cc426.mp4

**prompt:** `a palm tree, low poly 3d model`. Adan's model has a better shadow part.

https://user-images.githubusercontent.com/10042844/211014613-6373253d-7a37-4b66-ac1b-d04bb7819c01.mp4

https://user-images.githubusercontent.com/10042844/211014610-67817157-fe9e-4ace-a188-e84d88bf0f66.mp4
